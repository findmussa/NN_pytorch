import torch
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import nn_pytorch.config as config
from nn_pytorch.data import load_data, split_data
from nn_pytorch.models.dynamic_FNN import FNN
from nn_pytorch.plots import save_parity_plot
from nn_pytorch.utils.device import get_device

def main() -> None:
    DEVICE = get_device()
    print(f'Using device: {DEVICE}')

    # load checkpoint
    checkpoint = torch.load(config.MODEL_DIR/'checkpoint.pth', map_location=DEVICE)

    # load scalers
    scalers = joblib.load(config.MODEL_DIR/'scalers.pkl')
    sX = scalers['scaler_X']
    sY = scalers['scaler_Y']

    # data
    X, y = load_data(config.DATA_DIR/'data.csv')
    _, _, X_test, _, _, y_test = split_data(X, y, random_state=config.RANDOM_STATE)
    X_test_s = sX.transform(X_test)
    y_test_s = sY.transform(y_test.reshape(-1, 1))

    # load model
    mdl = FNN(checkpoint['in_feat'], checkpoint['out_feat'], checkpoint['hideen_layers'], checkpoint['activation']).to(DEVICE)
    mdl.load_state_dict(checkpoint['model_state_dict'])
    mdl.eval()
    print('model loaded successfully')

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
    with torch.inference_mode():
        preds_s = mdl(X_test_t).cpu().numpy()

    # inverse transform
    y_true = sY.inverse_transform(y_test_s).flatten()
    y_pred = sY.inverse_transform(preds_s).flatten()

    # metric
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f'  Test set evaluation')
    print(f"{'='*40}")
    print(f"  RMSE : {rmse:>12,.2f}")
    print(f"  MAE  : {mae:>12,.2f}")
    print(f"  R²   : {r2:>12.4f}")
    print(f"{'='*40}\n")

    # plots
    save_parity_plot(y_true, y_pred, config.FIG_DIR/'parity.pdf')
    print(f"plots saved to {config.FIG_DIR}")

if __name__ == "__main__":
    main()