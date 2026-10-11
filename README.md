# BudgetExit

Implementation of **BudgetExit: Budget-Conditioned Early Exiting with Dual Monotonicity Regularization**. A four-exit ResNet-18 receives a continuous compute budget and learns conditional stopping gates. Inference executes only the stages reached by each image.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and workflow

`prepare` downloads [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision and creates a seeded, class-balanced validation split of 5,000 training images. The remaining 45,000 images form the training pool; the official test split is untouched. The prepared directory contains `train.npz`, `val.npz`, `test.npz` and `meta.json`. Array indices identify the original training examples.

For [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), extract the archive to `data/tiny-imagenet-200`. Its annotated validation images are split deterministically into validation and test halves because the released test images have no labels. `prepare` writes image-path manifests and does not copy the images. A custom image collection can use `dataset="folder"` with identical class directories under `root/train`, `root/val` and `root/test`.

```bash
python -m budgetexit prepare --config configs/cifar100.json
python -m budgetexit train --config configs/cifar100.json --output outputs/budgetexit
python -m budgetexit evaluate --config configs/cifar100.json --checkpoint outputs/budgetexit/model.pt --output outputs/budgetexit
python -m budgetexit plot --config configs/cifar100.json --output outputs/budgetexit
```

Use `configs/tinyimagenet.json` for the Tiny-ImageNet workflow. Training reports validation accuracy without consulting test labels. Evaluation saves deterministic routing measurements at budgets 0.4, 0.6 and 0.8.

## Comparisons and ablations

```bash
python -m budgetexit compare --config configs/cifar100.json --output outputs/comparison
python -m budgetexit plot --config configs/cifar100.json --output outputs/comparison
```

`compare` trains every configured variant for every configured seed. It can take substantially longer than a single training run.

| Variant | Training and routing |
| --- | --- |
| `full` | Final-head training and full-depth inference |
| `branchynet` | Deep supervision; maximum-probability early exit |
| `entropy` | Deep supervision; normalized-entropy early exit |
| `bound_only` | Budget penalty with zero budget input |
| `budget_input` | Adds budget-conditioned gates |
| `depth` | Adds depth monotonicity |
| `budgetexit` | Adds budget-order monotonicity |

The confidence and entropy baselines calibrate a threshold separately for each budget on validation data. Calibration maximizes validation accuracy among candidates whose mean normalized cost is within the mapped budget. BudgetExit always uses the fixed gate threshold 0.5.

## Configuration and measurements

Commands read the supplied JSON configuration. `--data`, `--output`, `--device`, `--seed`, `--batch-size`, `--epochs`, `--max-steps` and `--limit` override individual fields. `--device auto` chooses CUDA when available. `max_steps=0` and `limit=0` mean no limit. `max_steps` stops optimization after the requested number of updates; it does not produce a trained benchmark model. For other fields use `--set KEY=JSON`, for example `--set lr=0.01` or `--set alpha=0.5`. String values require JSON quotes.

| Field | Meaning |
| --- | --- |
| `alpha` | Expected routed classification-loss weight, default 0.5 |
| `lambda_depth` | Penalty on an increase in true-class cross entropy at deeper exits, default 0.2 |
| `lambda_budget` | Squared excess of expected cost above the mapped budget, default 2.0 |
| `lambda_order` | Penalty when expected cost decreases as budget increases, default 0.5 |
| `budgets` | Deterministic evaluation operating points |
| `latency_samples`, `latency_warmup` | Timed batch-size-one requests and untimed warm-up requests |

Training computes exit probabilities as conditional stop probability times survival through earlier gates. The last exit is mandatory. Budget `b` maps to `c1 + b * (1-c1)`, where costs are normalized by the full multi-exit path. The route loss and expected-cost bound are differentiable surrogates; evaluation uses actual chosen exits.

`history.csv` contains training loss and validation accuracy. `metrics.csv` and `metrics.json` contain accuracy, GFLOPs, FLOPs reduction, latency, exit proportions, BVR, BTE, MPD, monotone ratio, ECE and BOV. BVR counts requests exceeding their mapped upper bound. BOV compares each budget with the immediately smaller configured budget; the first row has no BOV value. MPD and monotone ratio use true-class probabilities from all heads for diagnostics. `accuracy_flops.pdf` and `budget_control.pdf` visualize the measured operating points.

## Implementation conventions

Convolution and linear operations are counted as two FLOPs per multiply–accumulate; gate, embedding and preceding exit-head costs are included. Normalization and activation costs are not included. Latency uses the actual conditional path and synchronized device timing. Costs are computed from the configured image size rather than copied from the paper. CIFAR uses a 3×3 stride-one stem and no initial max-pool. Optimizer, epoch count and data split are explicit in the configuration.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
