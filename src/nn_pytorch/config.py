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

# model
HIDDEN_LAYERS: list[int] = [224, 96]
ACTIVATION: str = 'leaky_relu'

# traininig
LR: float =  4.715e-03
BATCH_SIZE: int = 128
EPOCHS: int = 500
PATIENCE: int = 20

# scheduler
LR_FACTOR: float = 0.1393
LR_PATIENCE: int = 11
LR_MIN:float = 1e-6   # minimum LR

# reproducubility
RANDOM_STATE: int= 1
