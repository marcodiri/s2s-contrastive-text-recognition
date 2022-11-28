import torch.nn as nn


class WindowToInstance(nn.Module):
    def __init__(self, out_instances):
        super().__init__()
        self.Mapping = nn.AdaptiveAvgPool2d((None, out_instances))
        self.Mapping_output = out_instances
    
    def forward(self, features):
        return self.Mapping(features)
