import torch
from torchmetrics import Metric


class DiceMetric(Metric):
    def __init__(self, classes: int, dist_sync_on_step: bool = False, smooth: float = 1e-8):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(DiceMetric, self).__init__()

        self._classes = classes
        self._smooth = smooth
        
        self.add_state('dice', default=torch.tensor([0.0 for _ in range(classes)]), dist_reduce_fx='sum')

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert inputs.shape == targets.shape
        
        # Binary threshold for sigmoid output
        inputs = (inputs > 0.5).float()
        
        # For binary segmentation, no need for argmax or one_hot
        targets = targets.float()
        
        # Flatten for calculation
        targets = torch.flatten(targets, 1)
        inputs = torch.flatten(inputs, 1)
        
        intersection = torch.sum(targets * inputs, dim=1)
        total = torch.sum(targets, dim=1) + torch.sum(inputs, dim=1)
        
        dice = (2. * intersection + self._smooth) / (total + self._smooth)

        self.dice += dice.mean(dim=0)

    def compute(self):
        return self.dice.mean().item()
