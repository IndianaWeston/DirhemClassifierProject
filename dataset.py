from collections import Counter
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from config import DATA_DIR, IMAGE_SIZE, TRAIN_SPLIT, BATCH_SIZE
from PIL import Image
from PIL import Image, UnidentifiedImageError

from PIL import Image, UnidentifiedImageError

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except UnidentifiedImageError:
        print(f"❌ UnidentifiedImageError at: {path}")
        return Image.new('RGB', (224, 224))
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return Image.new('RGB', (224, 224))



def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

from torchvision.datasets import ImageFolder

class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None, loader=None):
        super().__init__(root, transform=transform, loader=loader)

        # Filter out files starting with "._"
        self.samples = [
            (path, class_idx)
            for path, class_idx in self.samples
            if not os.path.basename(path).startswith("._")
        ]
        self.imgs = self.samples  # sync internal reference used by ImageFolder

def get_loaders():
    transform = get_transforms()

    train_path = os.path.join(DATA_DIR, 'Train')
    val_path = os.path.join(DATA_DIR, 'Val')

    train_ds = FilteredImageFolder(train_path, transform=transform, loader=pil_loader)
    val_ds   = FilteredImageFolder(val_path, transform=transform, loader=pil_loader)


    print("Train class_to_idx:", train_ds.class_to_idx)
    print("Val class_to_idx:", val_ds.class_to_idx)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    
    train_labels = [label for _, label in train_ds.samples]
    val_labels   = [label for _, label in val_ds.samples]

    print("Train class distribution:")
    for idx, count in Counter(train_labels).items():
        print(f"  {idx_to_class[idx]}: {count}")

    print("Val class distribution:")
    for idx, count in Counter(val_labels).items():
        print(f"  {idx_to_class[idx]}: {count}")

        print("🔁 get_loaders() CALLED")



    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    return train_loader, val_loader,





    