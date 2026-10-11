import argparse
import csv
import json
import random
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(config):
    value = config.get("device", "auto")
    return torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path, rows):
    if not rows:
        raise ValueError("No measurements to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_config(commands):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=commands)
    parser.add_argument("--config", required=True)
    for key in [
        "data",
        "output",
        "root",
        "checkpoint",
        "device",
        "variant",
        "protocol",
    ]:
        parser.add_argument("--" + key)
    for key in [
        "epochs",
        "batch-size",
        "max-steps",
        "limit",
        "seed",
        "workers",
        "stages",
    ]:
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key, value in vars(args).items():
        if key not in ["command", "config", "set"] and value is not None:
            config[key] = value
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in config:
            parser.error(f"Unknown configuration key: {key}")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            parser.error(f"Value for {key} must be JSON; quote strings as JSON strings")
    seed_everything(config.get("seed", 2026))
    torch.set_num_threads(config.get("threads", 4))
    return (args.command, config)


def balanced_limit(labels, limit, seed):
    if not limit or limit >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == k)) for k in np.unique(labels)]
    ordered = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    return np.asarray(ordered[:limit])


def prepare(config):
    root = Path(config.get("root", "data/raw"))
    out = Path(config["data"])
    out.mkdir(parents=True, exist_ok=True)
    name = config["dataset"]
    seed = config.get("seed", 2026)
    if name in ["cifar10", "cifar100"]:
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train = cls(str(root), train=True, download=True)
        test = cls(str(root), train=False, download=True)
        labels = np.asarray(train.targets)
        rng = np.random.default_rng(seed)
        validation = []
        count = int(config.get("val_size", 5000))
        if count < len(train.classes) or count >= len(labels):
            raise ValueError(
                "val_size must cover every class and be smaller than training size"
            )
        for k in range(len(train.classes)):
            validation.extend(
                rng.permutation(np.flatnonzero(labels == k))[
                    : count // len(train.classes)
                ]
            )
        val_idx = np.asarray(validation)
        train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)
        for split, indices in [("train", train_idx), ("val", val_idx)]:
            indices = indices[
                balanced_limit(labels[indices], config.get("limit", 0), seed)
            ]
            np.savez_compressed(
                out / f"{split}.npz",
                images=train.data[indices],
                labels=labels[indices],
                indices=indices,
            )
        test_idx = balanced_limit(
            np.asarray(test.targets), config.get("limit", 0), seed
        )
        np.savez_compressed(
            out / "test.npz",
            images=test.data[test_idx],
            labels=np.asarray(test.targets)[test_idx],
            indices=test_idx,
        )
        save_json(
            out / "meta.json", {"dataset": name, "classes": train.classes, "seed": seed}
        )
    elif name in ["tinyimagenet", "folder"]:
        manifest = {}
        train = datasets.ImageFolder(root / "train")
        classes = train.classes
        for split in ["train", "val", "test"]:
            if name == "tinyimagenet" and split in ["val", "test"]:
                annotations = root / "val" / "val_annotations.txt"
                mapping = {
                    line.split()[0]: line.split()[1]
                    for line in annotations.read_text().splitlines()
                }
                rows = [
                    (root / "val" / "images" / file, classes.index(label))
                    for file, label in sorted(mapping.items())
                ]
                rng = np.random.default_rng(seed)
                rng.shuffle(rows)
                midpoint = len(rows) // 2
                rows = rows[:midpoint] if split == "val" else rows[midpoint:]
            else:
                dataset = datasets.ImageFolder(root / split)
                if dataset.classes != classes:
                    raise ValueError(
                        "Class directory names must match between train, val and test"
                    )
                rows = dataset.samples
            indices = balanced_limit(
                np.asarray([y for _, y in rows]), config.get("limit", 0), seed
            )
            manifest[split] = [
                [str(Path(rows[i][0]).resolve()), int(rows[i][1])] for i in indices
            ]
        save_json(out / "manifest.json", manifest)
        save_json(
            out / "meta.json", {"dataset": name, "classes": classes, "seed": seed}
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


class Images(Dataset):
    def __init__(self, config, split, augment=False):
        root = Path(config["data"])
        self.size = config.get("image_size", 32)
        self.augment = augment
        self.rows = None
        if (root / "manifest.json").exists():
            self.rows = json.loads((root / "manifest.json").read_text())[split]
            self.labels = np.asarray([row[1] for row in self.rows])
        else:
            with np.load(root / f"{split}.npz", allow_pickle=False) as values:
                self.images = values["images"]
                self.labels = values["labels"]
        self.indices = balanced_limit(
            self.labels, config.get("limit", 0), config.get("seed", 2026)
        )
        self.labels = self.labels[self.indices]
        if self.rows is None:
            self.images = self.images[self.indices]
        else:
            self.rows = [self.rows[i] for i in self.indices]
        self.classes = json.loads((root / "meta.json").read_text())["classes"]
        operations = [transforms.Resize((self.size, self.size))]
        if augment:
            operations += [
                transforms.RandomCrop(self.size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        self.transform = transforms.Compose(operations + [transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def image(self, index):
        if self.rows is None:
            return Image.fromarray(self.images[index])
        with Image.open(self.rows[index][0]) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        return (self.transform(self.image(index)), int(self.labels[index]))


def loader(dataset, config, shuffle=False, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size or config.get("batch_size", 64),
        shuffle=shuffle,
        num_workers=config.get("workers", 0),
        drop_last=False,
    )


def normalize(x):
    mean = x.new_tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
    std = x.new_tensor([0.247, 0.2435, 0.2616])[None, :, None, None]
    return (x - mean) / std


def plot_csv(path, output, x, y, group="variant"):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV contains no measurements")
    figure, axis = plt.subplots(figsize=(7, 4))
    for label in sorted(set((row.get(group, "measurement") for row in rows))):
        subset = [row for row in rows if row.get(group, "measurement") == label]
        subset.sort(key=lambda row: float(row[x]))
        axis.plot(
            [float(row[x]) for row in subset],
            [float(row[y]) for row in subset],
            marker="o",
            label=label,
        )
    axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    axis.legend()
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
