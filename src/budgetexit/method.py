import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from .common import normalize

VARIANTS = [
    "budgetexit",
    "bound_only",
    "budget_input",
    "depth",
    "branchynet",
    "entropy",
    "full",
]


class BudgetExit(nn.Module):
    def __init__(self, classes, image_size=32):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.stages = nn.ModuleList(
            [base.layer1, base.layer2, base.layer3, base.layer4]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(channels, classes) for channels in [64, 128, 256, 512]]
        )
        self.budget_embedding = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(channels + 16, 32), nn.ReLU(), nn.Linear(32, 1))
                for channels in [64, 128, 256]
            ]
        )
        self.image_size = image_size

    def features_and_logits(self, x):
        x = self.stem(normalize(x))
        features, logits = [], []
        for stage, head in zip(self.stages, self.heads):
            x = stage(x)
            pooled = x.mean((2, 3))
            features.append(pooled)
            logits.append(head(pooled))
        return features, torch.stack(logits, dim=1)

    def stopping_probabilities(self, features, budget, conditioned=True):
        embedding = self.budget_embedding(
            budget[:, None] if conditioned else torch.zeros_like(budget[:, None])
        )
        conditional = [
            gate(torch.cat([feature, embedding], 1)).sigmoid().squeeze(1)
            for feature, gate in zip(features, self.gates)
        ]
        conditional.append(torch.ones_like(budget))
        conditional = torch.stack(conditional, 1)
        survival = torch.cumprod(
            torch.cat([torch.ones_like(budget[:, None]), 1 - conditional[:, :-1]], 1), 1
        )
        return conditional * survival

    @torch.no_grad()
    def route(self, x, budget, variant="budgetexit", threshold=0.5):
        batch = len(x)
        active = torch.arange(batch, device=x.device)
        output = x.new_empty(batch, self.heads[0].out_features)
        exits = torch.empty(batch, dtype=torch.long, device=x.device)
        budget_vector = x.new_full((batch, 1), float(budget))
        if variant == "bound_only":
            budget_vector.zero_()
        embedding = (
            self.budget_embedding(budget_vector)
            if variant not in ["branchynet", "entropy", "full"]
            else None
        )
        values = self.stem(normalize(x))
        for i, (stage, head) in enumerate(zip(self.stages, self.heads)):
            values = stage(values)
            if variant == "full" and i < 3:
                continue
            pooled = values.mean((2, 3))
            logits = head(pooled)
            if i == 3:
                stop = torch.ones(len(active), dtype=torch.bool, device=x.device)
            elif variant == "branchynet":
                stop = logits.softmax(1).max(1).values >= threshold
            elif variant == "entropy":
                p = logits.softmax(1)
                stop = (
                    -(p * p.clamp_min(1e-8).log()).sum(1)
                    / torch.log(x.new_tensor(p.shape[1]))
                    <= threshold
                )
            else:
                stop = (
                    self.gates[i](torch.cat([pooled, embedding[active]], 1))
                    .sigmoid()
                    .squeeze(1)
                    >= 0.5
                )
            output[active[stop]] = logits[stop]
            exits[active[stop]] = i
            active = active[~stop]
            if not len(active):
                break
            values = values[~stop]
        return output, exits

    @torch.no_grad()
    def cost_profile(self, variant):
        total = 0
        handles = []

        def count(module, inputs, output):
            nonlocal total
            if isinstance(module, nn.Conv2d):
                total += (
                    output.numel()
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.in_channels
                    // module.groups
                    * 2
                )
            elif isinstance(module, nn.Linear):
                total += output.numel() * module.in_features * 2

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count))
        training = self.training
        self.eval()
        x = torch.zeros(
            1,
            3,
            self.image_size,
            self.image_size,
            device=next(self.parameters()).device,
        )
        embedding = (
            self.budget_embedding(x.new_zeros(1, 1))
            if variant not in ["branchynet", "entropy", "full"]
            else None
        )
        x = self.stem(normalize(x))
        costs = []
        for i, (stage, head) in enumerate(zip(self.stages, self.heads)):
            x = stage(x)
            if variant != "full" or i == 3:
                feature = x.mean((2, 3))
                head(feature)
                if i < 3 and embedding is not None:
                    self.gates[i](torch.cat([feature, embedding], 1))
            costs.append(total)
        for handle in handles:
            handle.remove()
        self.train(training)
        return torch.tensor(costs, device=x.device, dtype=torch.float64)


def objective(model, x, y, config, costs):
    features, logits = model.features_and_logits(x)
    ce = torch.stack(
        [F.cross_entropy(logits[:, i], y, reduction="none") for i in range(4)], 1
    )
    variant = config["variant"]
    if variant == "full":
        return ce[:, -1].mean()
    head = ce.mean()
    if variant in ["branchynet", "entropy"]:
        return head
    budgets = torch.sort(torch.rand(len(x), 2, device=x.device), dim=1).values
    probabilities = [
        model.stopping_probabilities(
            features, budgets[:, i], conditioned=variant != "bound_only"
        )
        for i in range(2)
    ]
    normalized = costs.to(x.dtype) / costs[-1]
    expected = [(p * normalized).sum(1) for p in probabilities]
    targets = normalized[0] + budgets * (1 - normalized[0])
    route = sum((p * ce).sum(1).mean() for p in probabilities) / 2
    bound = (
        sum(F.relu(expected[i] - targets[:, i]).square().mean() for i in range(2)) / 2
    )
    depth = F.relu(ce[:, 1:] - ce[:, :-1]).mean()
    order = F.relu(expected[0] - expected[1]).mean()
    lambda_d = config["lambda_depth"] if variant in ["depth", "budgetexit"] else 0
    lambda_o = config["lambda_order"] if variant == "budgetexit" else 0
    return (
        head
        + config["alpha"] * route
        + config["lambda_budget"] * bound
        + lambda_d * depth
        + lambda_o * order
    )
