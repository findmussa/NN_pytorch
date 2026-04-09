import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

def load_data(data_path: str, target_col: str):
    df = pd.read_csv(data_path)
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    return X, y

def split_data(X, y, random_state: int = 1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state= random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    sX = StandardScaler().fit(X_train)
    sY = StandardScaler().fit(y_train.reshape(-1,1))

    X_train_s = sX.transform(X_train)
    X_val_s = sX.transform(X_val)
    X_test_s = sX.transform(X_test)

    y_train_s = sY.transform(y_train.reshape(-1,1))
    y_val_s = sY.transform(y_val.reshape(-1,1))
    y_test_s = sY.transform(y_test.reshape(-1,1))
    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, sX, sY

def make_loaders(X_train, X_val, y_train, y_val, batch_size: int, device: torch.device):
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
