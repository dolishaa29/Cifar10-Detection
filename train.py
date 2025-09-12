# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import os
from model import CIFARResNet18
from utils import train_transform, val_transform

def train(
    data_dir="./data",
    epochs=20,
    batch_size=128,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    device=None,
    save_path="saved_models/best_model.pth"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_ds   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CIFARResNet18(num_classes=10, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

        scheduler.step()

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch} validation accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Saved new best model to {save_path} (val_acc={val_acc:.2f}%)")

    print("Training finished. Best val acc:", best_acc)

if __name__ == "__main__":
    train(epochs=30, batch_size=128, lr=0.1, save_path="saved_models/best_model.pth")
