import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from config import DATA_DIR, IMAGE_SIZE, TRAIN_SPLIT, BATCH_SIZE

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_loaders():
    dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms())
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return (
        torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)
    )
