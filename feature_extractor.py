import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    Uses pretrained ResNet18 as fixed feature extractor.
    """

    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return f.view(f.size(0), -1)

