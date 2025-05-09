# File: local_code/stage_3_code/stage3.py
"""
Stage 3 CNN training script:
- Trains on MNIST, ORL (faces), CIFAR-10 datasets
- Uses Accuracy, Precision, Recall, F1 as evaluation metrics
- Saves learning curves and metrics plots under the specified result directory
- Prints a summary table of metrics per epoch at the end of each dataset run
- Allows separate epoch counts for CIFAR-10 via epochs_cifar parameter
- MNIST DataLoader uses multiple workers and pinned memory for speed
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1) Data loading helper
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 2) Dataset wrapper
class PickleDataset(Dataset):
    def __init__(self, data_list, gray=False):
        self.data = data_list
        self.gray = gray
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]['image']
        lbl = self.data[idx]['label']
        if self.gray and img.ndim == 3:
            img = img[:, :, 0]
            lbl -= 1  # ORL labels 1–40 → 0–39
        tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        return tensor, lbl

# 3) Model definitions
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*7*7, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))  # no dropout here
        return self.fc2(x)

class ORLCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*28*23, 256)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 40)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(128*8*8, 256)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# 4) Dataloaders setup
def get_dataloaders(data_dir, bs_mnist=64, bs_face=32, bs_cifar=64):
    paths = {k: os.path.join(data_dir, k) for k in ['MNIST','ORL','CIFAR']}
    mnist = load_pickle(paths['MNIST'])
    orl   = load_pickle(paths['ORL'])
    cifar = load_pickle(paths['CIFAR'])
    mnist_train = DataLoader(
        PickleDataset(mnist['train'], gray=True),
        batch_size=bs_mnist, shuffle=True,
        num_workers=4, pin_memory=True
    )
    mnist_test = DataLoader(
        PickleDataset(mnist['test'],  gray=True),
        batch_size=bs_mnist, shuffle=False,
        num_workers=4, pin_memory=True
    )
    orl_train = DataLoader(
        PickleDataset(orl['train'], gray=True),
        batch_size=bs_face, shuffle=True
    )
    orl_test = DataLoader(
        PickleDataset(orl['test'],  gray=True),
        batch_size=bs_face, shuffle=False
    )
    cifar_train = DataLoader(
        PickleDataset(cifar['train'], gray=False),
        batch_size=bs_cifar, shuffle=True
    )
    cifar_test = DataLoader(
        PickleDataset(cifar['test'],  gray=False),
        batch_size=bs_cifar, shuffle=False
    )
    return mnist_train, mnist_test, orl_train, orl_test, cifar_train, cifar_test

# 5) Training & evaluation
def train_model(model, train_loader, test_loader,
                epochs, lr, weight_decay, device, prefix, out_dir):
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)

    train_losses, test_losses = [], []
    metrics_history = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}

    for ep in range(1, epochs+1):
        # training
        model.train()
        running = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        # evaluation
        model.eval()
        all_preds, all_labels = [], []
        running, correct = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                running += loss.item() * X.size(0)
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                correct += (preds == y).sum().item()
        test_loss = running / len(test_loader.dataset)
        test_losses.append(test_loss)

        # compute metrics
        acc  = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        metrics_history['accuracy'].append(acc)
        metrics_history['precision'].append(prec)
        metrics_history['recall'].append(rec)
        metrics_history['f1'].append(f1)

        print(f"{prefix} Ep {ep}:")
        print(f"  Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
        print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # save plots & summary
    epochs_range = range(1, epochs+1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses,  label='Test Loss')
    plt.title(f'{prefix} Loss Curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(out_dir, f'{prefix}_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for m in metrics_history:
        plt.figure()
        plt.plot(epochs_range, metrics_history[m], label=m.capitalize())
        plt.title(f'{prefix} {m.capitalize()} Curve')
        plt.xlabel('Epoch'); plt.ylabel(m.capitalize()); plt.legend()
        plt.savefig(os.path.join(out_dir, f'{prefix}_{m}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n=== {prefix} Epoch-wise Metrics Summary ===")
    print("Epoch\tAccuracy\tPrecision\tRecall\tF1-score")
    for i in range(epochs):
        print(f"{i+1}\t{metrics_history['accuracy'][i]:.4f}\t"
              f"{metrics_history['precision'][i]:.4f}\t"
              f"{metrics_history['recall'][i]:.4f}\t"
              f"{metrics_history['f1'][i]:.4f}")

# 6) Main orchestration
def main_stage3(data_dir, result_dir, device_str,
                epochs=10, epochs_cifar=None,
                lr=1e-3, weight_decay=1e-4,
                bs_mnist=64, bs_face=32, bs_cifar=64):
    device = torch.device(device_str)
    mnist_tr, mnist_te, orl_tr, orl_te, cifar_tr, cifar_te = \
        get_dataloaders(data_dir, bs_mnist, bs_face, bs_cifar)

    train_model(MNISTCNN(), mnist_tr, mnist_te,
                epochs, lr, weight_decay, device, 'MNIST', result_dir)
    train_model(ORLCNN(), orl_tr, orl_te,
                epochs, lr, weight_decay, device, 'ORL_face', result_dir)

    cifar_epochs = epochs_cifar or epochs
    train_model(CIFARCNN(), cifar_tr, cifar_te,
                cifar_epochs, lr, weight_decay, device, 'CIFAR10', result_dir)
