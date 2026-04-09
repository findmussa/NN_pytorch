# Neural Network with PyTorch

A clean, reusable template for regression tasks using feedforward neural networks in PyTorch. Includes training, evaluation, and hyperparameter optimisation with Optuna.

---

## Project Structure

```text
nn-pytorch/
├── scripts/
│   ├── train.py            # training entry point
│   ├── evaluate.py         # evaluation entry point
│   └── optimize.py         # hyperparameter optimisation
├── src/nn_pytorch/
│   ├── config.py           # hyperparameters and target column
│   ├── data.py             # data loading, splitting, scaling
│   ├── trainer.py          # training and evaluation loops
│   ├── plots.py            # loss and parity plots
│   ├── models/
│   │   └── dynamic_FNN.py  # flexible feedforward neural network
│   └── utils/
│       ├── paths.py        # project root and directory paths
│       └── device.py       # device detection (cpu/mps/cuda)
├── data/                   # raw data (not tracked)
├── models/                 # saved checkpoints and scalers (not tracked)
├── figures/                # plots and outputs (not tracked)
├── notebooks/              # exploratory notebooks
├── pyproject.toml          # dependencies
└── uv.lock                 # locked dependencies
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Clone repository
git clone https://github.com/findmussa/NN_pytorch.git
cd NN_pytorch

# Install dependencies
uv sync
```

---

## Usage

### 1. Add Your Data

Place your dataset at:

```text
data/data.csv
```

---

### 2. Configure

Edit `src/nn_pytorch/config.py`:

```python
# data
TARGET_COL    = 'price'     # target column name in your CSV

# model
HIDDEN_LAYERS = [64, 32]    # neurons per hidden layer
ACTIVATION    = 'relu'      # relu, tanh, leaky_relu, elu, gelu

# training
EPOCHS        = 500
BATCH_SIZE    = 32
LR            = 1e-3
PATIENCE      = 20          # early stopping patience

# scheduler
LR_FACTOR     = 0.5         # reduce LR by this factor
LR_PATIENCE   = 10          # epochs before reducing LR
LR_MIN        = 1e-6        # minimum LR

# reproducibility
RANDOM_STATE  = 1
```

**Note:** For a new dataset, you typically only need to update `TARGET_COL`.

---

### 3. Train

```bash
uv run python scripts/train.py
```

Outputs saved to `models/`:

* `checkpoint.pth` — best model weights + architecture
* `scalers.pkl` — fitted preprocessing scalers

---

### 4. Evaluate

```bash
uv run python scripts/evaluate.py
```

Outputs:

* Console: test set metrics
* `figures/`:

  * `loss.pdf` — training vs validation loss
  * `parity.png` — predicted vs true values

---

### 5. Hyperparameter Optimisation (Optional)

```bash
uv run python scripts/optimize.py
```

Uses Optuna with:

* **TPE sampler**
* **Median pruner**

#### Search Space

| Parameter     | Range                             |
| ------------- | --------------------------------- |
| `n_layers`    | 1 → 4                             |
| `hidden_size` | 16 → 256 (per layer)              |
| `activation`  | relu, tanh, leaky_relu, elu, gelu |
| `lr`          | 1e-4 → 1e-2                       |
| `batch_size`  | 16, 32, 64, 128                   |
| `lr_factor`   | 0.1 → 0.5                         |
| `lr_patience` | 5 → 20                            |

After optimisation:

1. Update `config.py` with best parameters
2. Retrain the model

---

## Model

Flexible feedforward neural network:

```text
Input → [Linear → Activation] × n_layers → Linear → Output
```

* No activation on output layer (regression)
* Supported activations:

  * `relu`, `tanh`, `leaky_relu`, `elu`, `gelu`, `sigmoid`

---

## Path Resolution

* Paths are resolved automatically from the project root
* No hardcoded paths required
* Root is detected via the `data/` directory
* All required directories are created on first run

---

## Dependencies

| Package      | Purpose                     |
| ------------ | --------------------------- |
| PyTorch      | Deep learning framework     |
| NumPy        | Numerical computing         |
| Pandas       | Data handling               |
| scikit-learn | Preprocessing & metrics     |
| Matplotlib   | Plotting                    |
| torchinfo    | Model summary               |
| Optuna       | Hyperparameter optimisation |
| joblib       | Scaler serialisation        |

---

## Workflow

```text
optimize.py → find best hyperparameters
       ↓
config.py   → update with best params
       ↓
train.py    → train final model
       ↓
evaluate.py → metrics + plots
```

---

## Author

**Nur MM Kalimullah, PhD**
Research Fellow, Trinity College Dublin

* GitHub: https://github.com/findmussa
* Google Scholar: https://scholar.google.com/citations?user=yrrCtqwAAAAJ&hl=en
* Website: https://findmussa.github.io
