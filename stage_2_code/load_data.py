import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_csv(path: str):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y

def get_dataloaders(
    train_path: str,
    test_path:  str,
    batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    X_train, y_train = load_csv(train_path) # X is my train and y is my train_lable
    X_test,  y_test  = load_csv(test_path)
    # Used for debug 
    print(">>> Train labels:", np.unique(y_train), "counts:", np.bincount(y_train))
    print(">>> Test  labels:", np.unique(y_test),  "counts:", np.bincount(y_test))

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size
    )
    return train_loader, test_loader