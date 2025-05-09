# Updated 3.5.py with correct label shifting (MNIST unshifted, ORL shifted)

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# — 1) Dataset & loaders —
class PickleDataset(Dataset):
    def __init__(self, data_list, gray=False, shift_label=False):
        self.data = data_list
        self.gray = gray
        self.shift_label = shift_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]['image']
        lbl = self.data[idx]['label']
        # if grayscale and 3-channel image, drop to single channel
        if self.gray and img.ndim == 3:
            img = img[:, :, 0]
        # shift labels for ORL only
        if self.shift_label:
            lbl = lbl - 1
        # convert to tensor and normalize
        t = torch.tensor(img, dtype=torch.float32) / 255.0
        # ensure shape CxHxW
        if t.ndim == 2:
            t = t.unsqueeze(0)
        else:
            t = t.permute(2, 0, 1)
        return t, lbl

def get_loaders(data_dir, bs_mnist=64, bs_face=32, bs_cifar=64):
    # MNIST: gray-scale, no label shift
    mnist = pickle.load(open(os.path.join(data_dir, 'MNIST'), 'rb'))
    mnist_tr = DataLoader(
        PickleDataset(mnist['train'], gray=True, shift_label=False),
        batch_size=bs_mnist, shuffle=True
    )
    mnist_te = DataLoader(
        PickleDataset(mnist['test'],  gray=True, shift_label=False),
        batch_size=bs_mnist, shuffle=False
    )

    # ORL: gray-scale, shift 1–40 → 0–39
    orl = pickle.load(open(os.path.join(data_dir, 'ORL'), 'rb'))
    orl_tr = DataLoader(
        PickleDataset(orl['train'], gray=True, shift_label=True),
        batch_size=bs_face, shuffle=True
    )
    orl_te = DataLoader(
        PickleDataset(orl['test'],  gray=True, shift_label=True),
        batch_size=bs_face, shuffle=False
    )

    # CIFAR-10: color, no shift
    cifar = pickle.load(open(os.path.join(data_dir, 'CIFAR'), 'rb'))
    cifar_tr = DataLoader(
        PickleDataset(cifar['train'], gray=False, shift_label=False),
        batch_size=bs_cifar, shuffle=True
    )
    cifar_te = DataLoader(
        PickleDataset(cifar['test'],  gray=False, shift_label=False),
        batch_size=bs_cifar, shuffle=False
    )

    return {
        'MNIST': (mnist_tr,  mnist_te,  1, 28, 28, 10, 5),
        'ORL':   (orl_tr,    orl_te,    1,112, 92, 40, 10),
        'CIFAR': (cifar_tr,  cifar_te,  3, 32, 32, 10, 15)
    }

# — 2) Configurable CNN —
class BaseCNN(nn.Module):
    def __init__(self, in_ch, kernels, filters, fc_dim, use_bn,
                 in_h, in_w, num_classes):
        super().__init__()
        ks1, ks2 = kernels
        f1, f2   = filters

        self.conv1 = nn.Conv2d(in_ch, f1, ks1, padding=ks1//2)
        self.bn1   = nn.BatchNorm2d(f1) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(f1, f2, ks2, padding=ks2//2)
        self.bn2   = nn.BatchNorm2d(f2) if use_bn else nn.Identity()
        self.pool  = nn.MaxPool2d(2, 2)

        # compute flatten size after two pool ops
        ph = in_h // 2 // 2
        pw = in_w // 2 // 2
        flat = f2 * ph * pw

        self.fc1  = nn.Linear(flat, fc_dim)
        self.drop = nn.Dropout(0.5)
        self.fc2  = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# — 3) Model variants —
variants = {
    "baseline_model": {
        "kernels": [5,3], "filters": [32,64],
        "fc_dim": 256,   "use_bn": True
    },
    "third_conv_layer_without_adjustment": {
        "kernels": [3,3], "filters": [32,64],
        "fc_dim": 256,   "use_bn": True
    },
    "increased_width_layers": {
        "kernels": [3,3], "filters": [64,128],
        "fc_dim": 512,   "use_bn": True
    },
    "without_batch_normalization": {
        "kernels": [3,3], "filters": [32,64],
        "fc_dim": 256,   "use_bn": False
    },
    "large_kernel_sizes": {
        "kernels": [7,5], "filters": [32,64],
        "fc_dim": 256,   "use_bn": True
    },
}

# — 4) Train & evaluate —
def train_and_eval(cfg, loaders, device):
    train_loader, test_loader, in_ch, in_h, in_w, num_cls, epochs = loaders
    model = BaseCNN(
        in_ch=in_ch, kernels=cfg["kernels"], filters=cfg["filters"],
        fc_dim=cfg["fc_dim"], use_bn=cfg["use_bn"],
        in_h=in_h, in_w=in_w, num_classes=num_cls
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(X), y).backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return acc, prec, rec, f1

# — 5) Main sweep —
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.abspath(os.path.join(script_dir, "../../data/stage_3_data"))
    result_dir = os.path.abspath(os.path.join(script_dir, "../../result/stage_3_result"))
    os.makedirs(result_dir, exist_ok=True)

    loaders = get_loaders(data_dir)
    results = []

    for ds_name, ld in loaders.items():
        for var_name, cfg in variants.items():
            acc, prec, rec, f1 = train_and_eval(cfg, ld, device)
            results.append({
                "dataset":   ds_name,
                "variant":   var_name,
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1":        f1
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, "config_sweep.csv"), index=False)
    print(df)
