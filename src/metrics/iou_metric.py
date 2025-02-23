import torch
import torch.nn.functional as F
from torchmetrics import Metric
import monai.metrics


class IOUMetric(Metric):
    def __init__(self, classes, dist_sync_on_step: bool = False, smooth: float = 1e-8):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        super(IOUMetric, self).__init__()

        self._classes = classes
        self._smooth = smooth
        
        self.add_state('iou', default=torch.tensor([0.0 for _ in range(classes)]), dist_reduce_fx='sum')

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert inputs.shape == targets.shape
        
        # Binary threshold for sigmoid output
        inputs = (inputs > 0.7).float()
        
        # For binary segmentation, no need for argmax or one_hot
        targets = targets.float()
        
        # Flatten for calculation
        targets = torch.flatten(targets, 1)
        inputs = torch.flatten(inputs, 1)
        
        intersection = torch.sum(targets * inputs, dim=1)
        union = torch.sum(targets, dim=1) + torch.sum(inputs, dim=1) - intersection
        
        iou = (intersection + self._smooth) / (union + self._smooth)

        self.iou += iou.mean(dim=0)

    def compute(self):
        return self.iou.mean().item()
