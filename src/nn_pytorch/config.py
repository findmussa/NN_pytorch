import sys
from pathlib import Path

def _find_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent/'data').exists():
            return parent
    
    raise RuntimeError("Project root not found")

ROOT: Path = _find_root()

# paths
DATA_DIR = ROOT/'data'
MODEL_DIR = ROOT / 'models'
FIG_DIR = ROOT/ 'figures'

# create dirs automatically on import
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# training
EPOCHS: int = 500
BATCH_SIZE: int = 32
LR: float = 1e-3
PATIENCE: int = 20

# model
H1: int = 64
H2: int = 32

# reproducubility
RANDOM_STATE: int= 1
