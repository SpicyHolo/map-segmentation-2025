import monai.losses
import torch
import torch.nn as nn


class FocalDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalDiceLoss, self).__init__()

        self.loss = monai.losses.DiceFocalLoss(
                include_background=False,
                reduction='none',
                softmax=False,
            )


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(inputs, targets)

        return torch.mean(loss)
