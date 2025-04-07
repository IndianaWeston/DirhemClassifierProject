import torch
from torch import nn, optim
from config import NUM_EPOCHS, LR
from model import get_model
from dataset import get_loaders
from utils import compute_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_loaders()
model = get_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train():
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

        validate()

def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train()
    # Save model after training
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

