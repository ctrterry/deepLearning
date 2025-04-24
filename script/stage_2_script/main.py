import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report

# relative imports within stage_2_code
from local_code.stage_2_code.load_data    import get_dataloaders, load_csv
from local_code.stage_2_code.method_MLP   import MLP
from local_code.stage_2_code.train_utils  import train_epoch
from local_code.stage_2_code.evaluation_accuracy   import (
    evaluate_loss,
    get_predictions_and_labels,
    compute_metrics
)
from local_code.stage_2_code.plot_utils import plot_loss_curves, plot_accuracy

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
TRAIN_PATH    = '/Users/terrychen/Desktop/ECS189G_Winter_2025_Source_Code_Template/data/stage_2_data/train.csv'
TEST_PATH     = '/Users/terrychen/Desktop/ECS189G_Winter_2025_Source_Code_Template/data/stage_2_data/test.csv'
batch_size    = 64
hidden_dims   = [128,32] # [256,128,64,32]
learning_rate = 1e-3
weight_decay  = 0 #1e-4
dropout       = 0.5
max_epochs    = 150
patience      = 10 # Allow up to 【10 。50. 100】 bad epochs before the stopping the traning. I'm trying to saving time. 
result_folder = '/Users/terrychen/Desktop/ECS189G_Winter_2025_Source_Code_Template/result/stage_2_result'
# result_file   = 'stage2_metrics'

# ─── SETUP ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_dataloaders(TRAIN_PATH, TEST_PATH, batch_size)
input_dim  = load_csv(TRAIN_PATH)[0].shape[1]
output_dim = len(np.unique(load_csv(TRAIN_PATH)[1]))

model     = MLP(input_dim, hidden_dims, output_dim, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# ─── TRAINING w/ EARLY STOPPING ───────────────────────────────────────────
train_losses, test_losses, train_accs = [], [], []
best_val_loss, counter = float('inf'), 0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, max_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_loss(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    test_losses.append(val_loss)
    train_accs.append(train_acc)

    # I'm trying to control overfitting issues.
    # I will compared with the current validation loss (val_loss) to the best validation loss has seen so far (best_val_loss)
    # 
    if val_loss < best_val_loss:
        best_val_loss = val_loss # Keep tracking the best perforamnce model on the validation 
        best_weights  = copy.deepcopy(model.state_dict())
        counter = 0     # if no improvment and then reset the patience counter to 0. 
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_weights)
            break

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:02d} → train_loss: {train_loss:.4f}, "
            f"test_loss: {val_loss:.4f}, train_acc: {train_acc:.4f}"
        )

# ─── PLOTTING ─────────────────────────────────────────────────────────────────
loss_curve_path = os.path.join(result_folder, 'loss_curve.png')
acc_curve_path  = os.path.join(result_folder, 'accuracy_curve.png')
plot_loss_curves(train_losses, test_losses, save_path=loss_curve_path)
plot_accuracy(train_accs, save_path=acc_curve_path)

# ─── MODEL PERFORMANCE ────────────────────────────────────────────────────────
# Gather metrics

y_true_train, y_pred_train = get_predictions_and_labels(model, train_loader, device)
metrics_train = compute_metrics(y_true_train, y_pred_train)

y_true_test, y_pred_test   = get_predictions_and_labels(model, test_loader, device)
metrics_test  = compute_metrics(y_true_test, y_pred_test)

print("\nTrain Performance:")
for k, v in metrics_train.items(): print(f"  {k.capitalize():>9}: {v:.4f}")
print("\nTest Performance:")
for k, v in metrics_test.items():  print(f"  {k.capitalize():>9}: {v:.4f}")

# Optional: detailed classification report
print("\nFull Test Set Report:\n")
print(classification_report(y_true_test, y_pred_test, digits=4))