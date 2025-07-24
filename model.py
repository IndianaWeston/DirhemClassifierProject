import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, MODEL_NAME

def get_model():
   from torchvision.models import resnet18, ResNet18_Weights
   model = resnet18(weights=ResNet18_Weights.DEFAULT)
   model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
   return model
