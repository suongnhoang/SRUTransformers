import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, num_class, num_feature, clamp_min=1e-12, clamp_max=1e+12):
        super(CenterLoss, self).__init__()
        self.clamp_min, self.clamp_max = clamp_min, clamp_max
        self.centers = nn.Parameter(torch.randn(num_class, num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=self.clamp_min, max=self.clamp_max).mean(dim=-1)
        return loss


class MultiSofmaxLoss(nn.Module):
    def __init__(self, class_n) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.class_n = class_n

    def forward(self, input, target):
        output = self.loss(input.reshape(-1, self.class_n), target.reshape(-1))
        return output

class MultiBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        output = self.loss(input.reshape(-1), target.reshape(-1))
        return output