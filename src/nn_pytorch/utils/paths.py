from pathlib import Path

def _find_root() -> Path:
    """this function find root of the project

    Raises:
        RuntimeError: Raises project root not found

    Returns:
        Path: Path of project root directory
    """
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent/'data').exists():
            return parent
        
    raise RuntimeError("Project Root Not Found")

ROOT_DIR: Path = _find_root()

# Directories
DATA_DIR: Path = ROOT_DIR/ "data"
MODELS_DIR: Path = ROOT_DIR / "models"
FIG_DIR: Path = ROOT_DIR/"figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)