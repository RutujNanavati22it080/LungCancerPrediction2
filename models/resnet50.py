import torchvision.models as models
import torch.nn as nn

def build_model(num_classes=2):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

