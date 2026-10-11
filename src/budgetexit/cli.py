from pathlib import Path
import json
import time
import numpy as np
import torch

from .common import (
    Images,
    loader,
    device_of,
    seed_everything,
    prepare,
    parse_config,
    save_json,
    save_rows,
    plot_csv,
)
from .method import BudgetExit, objective, VARIANTS


def train(config):
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    seed_everything(config["seed"])
    data = Images(config, "train", augment=True)
    model = BudgetExit(len(data.classes), config["image_size"]).to(device_of(config))
    costs = model.cost_profile(config["variant"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    history = []
    steps = 0
    for epoch in range(config["epochs"]):
        model.train()
        loss_sum = samples = 0
        for x, y in loader(data, config, shuffle=True):
            x, y = x.to(device_of(config)), y.to(device_of(config))
            loss = objective(model, x, y, config, costs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            samples += len(y)
            steps += 1
            if config["max_steps"] and steps >= config["max_steps"]:
                break
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader(Images(config, "val"), config):
                logits = model.features_and_logits(x.to(device_of(config)))[1][:, -1]
                correct += (logits.argmax(1).cpu() == y).sum().item()
                total += len(y)
        row = {
            "variant": config["variant"],
            "epoch": epoch + 1,
            "loss": loss_sum / samples,
            "val_accuracy": 100 * correct / total,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if config["max_steps"] and steps >= config["max_steps"]:
            break
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "classes": data.classes, "config": config},
        output / "model.pt",
    )
    save_rows(output / "history.csv", history)
    save_json(output / "config.json", config)
    return str(output / "model.pt")


@torch.no_grad()
def cache_logits(model, dataset, config):
    logits, labels = [], []
    for x, y in loader(dataset, config):
        logits.append(model.features_and_logits(x.to(device_of(config)))[1].cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def calibrate(logits, labels, costs, budget, variant):
    probabilities = logits.softmax(-1)
    scores = (
        probabilities.max(-1).values
        if variant == "branchynet"
        else -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        / np.log(probabilities.shape[-1])
    )
    normalized = costs.cpu() / costs[-1].cpu()
    target = normalized[0] + budget * (1 - normalized[0])
    candidates = []
    for threshold in torch.linspace(0, 1, 101):
        passed = scores >= threshold if variant == "branchynet" else scores <= threshold
        passed[:, -1] = True
        exits = passed.long().argmax(1)
        predictions = probabilities[torch.arange(len(labels)), exits].argmax(1)
        cost = normalized[exits].mean().item()
        acc = (predictions == labels).float().mean().item()
        candidates.append(
            (cost <= target, acc, -abs(cost - target.item()), float(threshold))
        )
    feasible = [item for item in candidates if item[0]]
    return max(
        feasible or candidates,
        key=lambda item: (item[1], item[2]) if feasible else (item[2], item[1]),
    )[-1]


def expected_calibration_error(probabilities, labels):
    confidence, prediction = probabilities.max(1)
    hits = prediction == labels
    value = 0.0
    for i in range(15):
        selected = (confidence >= i / 15) & (
            (confidence < (i + 1) / 15) if i < 14 else confidence <= 1
        )
        if selected.any():
            value += selected.float().mean().item() * abs(
                confidence[selected].mean().item()
                - hits[selected].float().mean().item()
            )
    return 100 * value


@torch.no_grad()
def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    variant = state["config"]["variant"]
    model = (
        BudgetExit(len(state["classes"]), state["config"]["image_size"])
        .to(device_of(config))
        .eval()
    )
    model.load_state_dict(state["model"])
    costs = model.cost_profile(variant)
    model.eval()
    test = Images(config, "test")
    all_logits, all_labels = cache_logits(model, test, config)
    probabilities = all_logits.softmax(-1)
    true_probabilities = probabilities.gather(
        2, all_labels[:, None, None].expand(-1, 4, 1)
    ).squeeze(-1)
    decrease = (
        torch.stack(
            [
                torch.relu(true_probabilities[:, i] - true_probabilities[:, j])
                for i in range(4)
                for j in range(i + 1, 4)
            ],
            1,
        )
        .max(1)
        .values.mean()
        .item()
    )
    monotone = (
        (true_probabilities[:, 1:] >= true_probabilities[:, :-1])
        .all(1)
        .float()
        .mean()
        .item()
    )
    val_logits, val_labels = (
        cache_logits(model, Images(config, "val"), config)
        if variant in ["branchynet", "entropy"]
        else (None, None)
    )
    rows, previous_exits = [], None
    normalized = costs.cpu() / costs[-1].cpu()
    for budget in sorted(config["budgets"]):
        threshold = (
            calibrate(val_logits, val_labels, costs, budget, variant)
            if val_logits is not None
            else 0.5
        )
        logits, exits = [], []
        for x, _ in loader(test, config):
            output, chosen = model.route(
                x.to(device_of(config)), budget, variant, threshold
            )
            logits.append(output.cpu())
            exits.append(chosen.cpu())
        logits, exits = torch.cat(logits), torch.cat(exits)
        realized = normalized[exits]
        target = normalized[0] + budget * (1 - normalized[0])
        timings = []
        count = min(config["latency_samples"], len(test))
        for i in range(count + config["latency_warmup"]):
            x = test[i % len(test)][0][None].to(device_of(config))
            if x.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.route(x, budget, variant, threshold)
            if x.is_cuda:
                torch.cuda.synchronize()
            if i >= config["latency_warmup"]:
                timings.append((time.perf_counter() - start) * 1000)
        row = {
            "variant": variant,
            "seed": state["config"]["seed"],
            "budget": budget,
            "threshold": threshold,
            "accuracy": 100 * (logits.argmax(1) == all_labels).float().mean().item(),
            "gflops": costs.cpu()[exits].mean().item() / 1e9,
            "flops_reduction": 100 * (1 - realized.mean().item()),
            "latency_ms": float(np.mean(timings)),
            "average_exit": exits.float().mean().item() + 1,
            "bvr": 100 * (realized > target + 1e-9).float().mean().item(),
            "bte": (realized - target).abs().mean().item(),
            "mpd": decrease,
            "monotone_ratio": monotone,
            "ece": expected_calibration_error(logits.softmax(1), all_labels),
            "bov": 100 * (previous_exits > exits).float().mean().item()
            if previous_exits is not None
            else None,
        }
        row.update(
            {
                f"exit_{i + 1}_ratio": (exits == i).float().mean().item()
                for i in range(4)
            }
        )
        rows.append(row)
        previous_exits = exits
    save_rows(Path(config["output"]) / "metrics.csv", rows)
    save_json(Path(config["output"]) / "metrics.json", rows)
    print(json.dumps(rows), flush=True)
    return rows


def compare(config):
    rows = []
    for seed in config["seeds"]:
        for variant in config["variants"]:
            current = dict(
                config,
                variant=variant,
                seed=seed,
                output=str(Path(config["output"]) / variant / str(seed)),
            )
            current["checkpoint"] = train(current)
            rows.extend(evaluate(current))
    save_rows(Path(config["output"]) / "metrics.csv", rows)


def main():
    command, config = parse_config(["prepare", "train", "evaluate", "compare", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        plot_csv(
            root / "metrics.csv", root / "accuracy_flops.pdf", "gflops", "accuracy"
        )
        plot_csv(root / "metrics.csv", root / "budget_control.pdf", "budget", "bvr")
