import monai.losses
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        

        self.loss = monai.losses.FocalLoss(
                include_background=False,
            )
        

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(inputs, targets)

        return torch.mean(loss)
