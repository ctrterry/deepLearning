import matplotlib.pyplot as plt
import os

def plot_loss_curves(
    train_losses: list[float],
    test_losses:  list[float],
    save_path:    str = None
) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, test_losses,  marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_accuracy(
    train_accs: list[float],
    save_path:  str = None
) -> None:
    epochs = range(1, len(train_accs) + 1)
    plt.figure()
    plt.plot(epochs, train_accs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()