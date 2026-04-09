
import torch
import torch.nn as nn
from torchinfo import summary
import joblib

from nn_pytorch.data import load_data, split_data, scale_data, make_loaders
import nn_pytorch.config as config
from nn_pytorch.utils.paths import MODELS_DIR, DATA_DIR, FIG_DIR
from nn_pytorch.models.dynamic_FNN import FNN
from nn_pytorch.trainer import train_one_epoch, evaluate
from nn_pytorch.utils.device import get_device
from nn_pytorch.plots import save_loss_plot


def main() -> None:
    torch.manual_seed(config.RANDOM_STATE)
    DEVICE = get_device()
    print(f'Using device: {DEVICE}')

    X, y = load_data(DATA_DIR/'data.csv', target_col=config.TARGET_COL)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=config.RANDOM_STATE)
    X_train_s, X_val_s, _, y_train_s, y_val_s, _, sX, sY = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
    train_loader, val_loader = make_loaders(X_train_s, X_val_s, y_train_s, y_val_s, batch_size=config.BATCH_SIZE, device=DEVICE)

    scalers = {'scaler_X': sX,
               'scaler_Y': sY}
    joblib.dump(scalers, MODELS_DIR/'scalers.pkl')

    in_feat = X_train_s.shape[1]
    out_feat = y_train_s.shape[1]


    mdl = FNN(in_feat, out_feat, config.HIDDEN_LAYERS, config.ACTIVATION).to(DEVICE)
    summary(mdl, input_size=(config.BATCH_SIZE, in_feat), device=DEVICE)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(mdl.parameters(), lr=config.LR)

    # add scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',                # minimize val loss
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN,
    )
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(mdl, train_loader, criterion, optimiser)
        val_loss = evaluate(mdl, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]['lr']    
    
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss     
            counter = 0
            checkpoint = {
                'model_state_dict': mdl.state_dict(),
                'in_feat': in_feat,
                'out_feat': out_feat,
                'hideen_layers': config.HIDDEN_LAYERS,
                'activation': config.ACTIVATION}
            torch.save(checkpoint, MODELS_DIR/'checkpoint.pth')
        else:
            counter += 1
            if counter >= config.PATIENCE:
                print(f"Early stopping at epoch: {epoch}")
                break

    save_loss_plot(train_losses, val_losses, FIG_DIR/'loss.pdf')
    print(f"plots saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
