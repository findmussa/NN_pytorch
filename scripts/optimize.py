import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler

# local import 
from nn_pytorch.data import load_data, split_data, scale_data, make_loaders
from nn_pytorch.models.dynamic_FNN import FNN
from nn_pytorch.trainer import train_one_epoch, evaluate
import nn_pytorch.config as config
from nn_pytorch.utils.device import get_device

DEVICE = get_device()
# data
X, y = load_data(config.DATA_DIR/'data.csv')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state= config.RANDOM_STATE)
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, sX, sY = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)

def objective(trial: optuna.Trial) -> float:

    # number of layers
    n_layers = trial.suggest_int('n_layers', 1, 4)

    # size of each layer
    hidden_layers = [trial.suggest_int(f'h{i}', 16, 256, step=16) for i in range(n_layers)]

    # activation function
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'sigmoid'])

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    lr_factor = trial.suggest_float('lr_factor', 0.1, 0.5)
    lr_patience = trial.suggest_int('lr_patience', 5, 20)

    # loaders
    train_loader, val_loader = make_loaders(X_train_s, X_val_s, y_train_s, y_val_s, batch_size= batch_size, device=DEVICE)
    # model
    in_feat = X_train_s.shape[1]
    out_feat = y_train_s.shape[1]
    mdl = FNN(in_feat, out_feat, hidden_layers, activation).to(DEVICE)

    # training setup
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(mdl.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,mode='min', factor=lr_factor, patience=lr_patience, min_lr=1e-6)

    # training loop
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(mdl, train_loader, criterion, optimiser)
        val_loss = evaluate(mdl, val_loader, criterion)
        scheduler.step(val_loss)

        # optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= config.PATIENCE:
                break
    
    return best_val_loss

def main():
    sampler = TPESampler(seed=config.RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)

    study = optuna.create_study(direction='minimize', sampler= sampler, pruner=pruner, study_name='fnn_optimisation')
    study.optimize(objective, n_trials=50,show_progress_bar=True)

    # results
    print(f"\n{'='*40}")
    print(f"  Optimization Results")
    print(f"{'='*40}")
    print(f"  Best val loss : {study.best_value:.4f}")
    print(f"  Best params   :")
    for k, v in study.best_params.items():
        print(f"    {k:15} : {v}")
    print(f"{'='*40}\n")


    # best architecture
    best = study.best_params
    n = best['n_layers']

    print('Update config file: ___')
    print("# model")
    print(f"HIDDEN_LAYERS: list[int] = {[best[f'h{i}'] for i in range(n)]}")
    print(f"ACTIVATION: str = '{best['activation']}'")
    print("\n# traininig")
    print(f"LR: float = {best['lr']: .3e}")
    print(f"BATCH_SIZE: int = {best['batch_size']}")
    print("\n# scheduler")
    print(f"LR_FACTOR: float = {best['lr_factor']:.4f}")
    print(f"LR_PATIENCE: int = {best['lr_patience']}")

if __name__ == "__main__":
    main()
