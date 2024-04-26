import torch
import torch.nn as nn

class MSELossCorr(nn.MSELoss):
    def __init__(self, regularization_strength=0.1, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.regularization_strength = regularization_strength

    def forward(self, input, target):
        mse_loss = super().forward(input, target)

        input_centered = input - input.mean()
        target_centered = target - target.mean()
        cov = (input_centered * target_centered).mean()
        input_variance = input_centered.pow(2).mean()
        target_variance = target_centered.pow(2).mean()
        correlation_coefficient = cov / (torch.sqrt(input_variance) * torch.sqrt(target_variance) + 1e-8)
        pearson_regularization = (1 - correlation_coefficient.pow(2))

        total_loss = mse_loss + self.regularization_strength * pearson_regularization
        return total_loss