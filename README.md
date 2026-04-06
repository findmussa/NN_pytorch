# Neural Network with PyTorch

A clean, professional template for regression tasks using feedforward neural networks in PyTorch. Includes training, evaluation, and hyperparameter optimisation with Optuna.

## Project Structure

```
nn-pytorch/
├── scripts/
│   ├── train.py         # training entry point
│   ├── evaluate.py      # evaluation entry point
│   └── optimize.py      # hyperparameter optimisation
├── src/nn_pytorch/
│   ├── config.py        # paths and hyperparameters
│   ├── data.py          # data loading, splitting, scaling
│   ├── trainer.py       # training and evaluation loops
│   ├── plots.py         # loss and parity plots
│   ├── models/
│   │   └── dynamic_FNN.py  # flexible feedforward neural network
│   └── utils/
│       └── device.py    # device detection (cpu/mps/cuda)
├── data/                # raw data (not tracked)
├── models/              # saved checkpoints and scalers (not tracked)
├── figures/             # plots and outputs (not tracked)
├── notebooks/           # exploratory notebooks
├── pyproject.toml       # dependencies
└── uv.lock              # locked dependencies
```

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

​```bash
# clone
git clone https://github.com/findmussa/NN_pytorch.git
cd NN_pytorch

# install dependencies
uv sync
​```

## Usage

### 1. Add Your Data
```bash
# place your CSV file at
data/data.csv
```

The target column is `price` by default. Change in `src/nn_pytorch/data.py`:
```python
X = df.drop('price', axis=1).values
y = df['price'].values
```

### 2. Configure

Edit `src/nn_pytorch/config.py`:
```python
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
LR_PATIENCE   = 10          # epochs to wait before reducing LR
LR_MIN        = 1e-6        # minimum LR

# reproducibility
RANDOM_STATE  = 1
```

### 3. Train
```bash
uv run python scripts/train.py
```

Saves to `models/`:
- `checkpoint.pth` — best model weights + architecture
- `scalers.pkl` — fitted scalers

### 4. Evaluate
```bash
uv run python scripts/evaluate.py
```

Prints test set metrics and saves to `figures/`:
- `loss.pdf` — training and validation loss curves
- `parity.png` — predicted vs true values

### 5. Hyperparameter Optimisation (Optional)
```bash
uv run python scripts/optimize.py
```

Optimises using Optuna with TPE sampler and median pruner. Search space:

| Parameter | Range |
|---|---|
| `n_layers` | 1 → 4 |
| `hidden_size` per layer | 16 → 256 |
| `activation` | relu, tanh, leaky_relu, elu, gelu |
| `lr` | 1e-4 → 1e-2 |
| `batch_size` | 16, 32, 64, 128 |
| `lr_factor` | 0.1 → 0.5 |
| `lr_patience` | 5 → 20 |

After optimisation, update `config.py` with the suggested values and retrain.

## Model

Flexible feedforward neural network with configurable depth and activation:
Input → [Linear → Activation] × n_layers → Linear → Output

Supports: `relu`, `tanh`, `leaky_relu`, `elu`, `gelu`, `sigmoid`

## Dependencies

| Package | Purpose |
|---|---|
| PyTorch | deep learning framework |
| NumPy | numerical computing |
| Pandas | data loading |
| scikit-learn | preprocessing, metrics |
| Matplotlib | plotting |
| torchinfo | model summary |
| Optuna | hyperparameter optimisation |
| joblib | scaler serialisation |

## Workflow

optimize.py → find best hyperparameters
↓
config.py   → update with best params
↓
train.py    → train final model
↓
evaluate.py → metrics + plots

## Author

**Nur MM Kalimullah, PhD** — Research Fellow, Trinity College Dublin  
GitHub: [@findmussa](https://github.com/findmussa)  
Google Scholar: [Nur MM Kalimullah](https://scholar.google.com/citations?user=yrrCtqwAAAAJ&hl=en)
Web: (https://findmussa.github.io)