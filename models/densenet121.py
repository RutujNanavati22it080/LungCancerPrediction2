import torchvision.models as models
import torch.nn as nn

def build_model(num_classes=2):
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
