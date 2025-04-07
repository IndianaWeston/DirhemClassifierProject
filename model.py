import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, MODEL_NAME

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
