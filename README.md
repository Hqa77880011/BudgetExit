# BudgetExit

Implementation of **BudgetExit: Budget-Conditioned Early Exiting with Dual Monotonicity Regularization**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves a class-balanced validation set and preserves the test split. It writes `train.npz`, `val.npz`, `test.npz` and `meta.json`. For [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), extract to `data/tiny-imagenet-200` and use `configs/tinyimagenet.json`.

```bash
python -m budgetexit prepare --config configs/cifar100.json
```

## Training and execution

`configs/cifar100.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m budgetexit train --config configs/cifar100.json --output outputs/budgetexit
```

## Evaluation

```bash
python -m budgetexit evaluate --config configs/cifar100.json --checkpoint outputs/budgetexit/model.pt --output outputs/budgetexit
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m budgetexit compare --config configs/cifar100.json --output outputs/comparison
python -m budgetexit plot --config configs/cifar100.json --output outputs/comparison
```

## Outputs

Run directories contain `model.pt`, `history.csv`, `metrics.csv`, `metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/metrics.csv`; plotting reads those computed files.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training and full evaluation were not run for this review. The GitHub Actions workflow checks packaging and source code.
