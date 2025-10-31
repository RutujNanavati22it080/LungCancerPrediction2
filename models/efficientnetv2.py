# models/efficientnetv2.py

from torchvision.models import efficientnet_v2_s
import torch.nn as nn

def build_model(num_classes):
    model = efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
