import torch
from torch import nn, optim
from config import NUM_EPOCHS, LR
from model import get_model
from dataset import get_loaders
from utils import compute_accuracy

import os
import sys

if os.environ.get("TRAIN_SCRIPT_RUNNING"):
    print(" Already running. Preventing re-entry.")
    sys.exit()

os.environ["TRAIN_SCRIPT_RUNNING"] = "1"
print(" START OF TRAIN.PY EXECUTION")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = get_loaders()
model = get_model().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
n_non_perforated = 900
n_perforated = 100
class_weights = torch.tensor([
    1.0 / n_non_perforated,
    1.0 / n_perforated
], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


def train():
        
    print("🧠 Starting training loop...")

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        print(f"Total training batches: {len(train_loader)}")

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = compute_accuracy(outputs, labels)
            correct += acc * labels.size(0)
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"\n Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f" Train Loss: {avg_loss:.4f} | Accuracy: {train_acc:.2f}%")

        val_acc = validate()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f" Saved new best model with val acc: {best_val_acc:.2f}%")

def validate():
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    print("\n Validating...")

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            acc = compute_accuracy(outputs, labels)
            correct += acc * labels.size(0)
            total += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    acc = 100 * correct / total if total > 0 else 0
    print(f" Val Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    print(" Returning train_loader and val_loader")

    return acc

if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


