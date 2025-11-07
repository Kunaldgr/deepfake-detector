import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # ResNeXt50 backbone
        model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(model.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x