import torchvision.models as models
import torch.nn as nn

def build_model(num_classes=2):
    model = models.efficientnet_b4(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
