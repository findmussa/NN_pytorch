import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_loss_plot(train_losses: list, val_losses: list, out_path: Path) -> None:
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label = 'Train')
    plt.plot(val_losses, label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([lo, hi], [lo, hi], '--', label = '1:1 plot')
    plt.xlabel("True Price")
    plt.ylabel("Pred Price")
    plt.legend()
    plt.title('Parity plot')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

