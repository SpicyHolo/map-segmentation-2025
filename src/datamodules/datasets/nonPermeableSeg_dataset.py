from typing import List, Tuple, Dict
from albumentations import Compose
import cv2
import numpy as np
from pathlib import Path
import torch
import json
import os

from torch.utils.data import Dataset


class nonPermeableSegDataset(Dataset):
    def __init__(self,
                 data_root: Path,
                 images_list: List,
                 augmentations: Compose,
                 coco_paths: List[Path]  # Dodano listę ścieżek do plików COCO
                 ):

        self._data_root = str(data_root)
        self._images_list = images_list
        self._augmentations = augmentations
        
        # Load COCO annotations from multiple files
        self.coco_data = []
        for coco_path in coco_paths:
            with open(coco_path, 'r') as f:
                self.coco_data.append(json.load(f))
            
        # Create image_id to annotations mapping
        self.image_to_masks: Dict[int, List] = {}
        for coco in self.coco_data:
            for ann in coco['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_to_masks:
                    self.image_to_masks[img_id] = []
                self.image_to_masks[img_id].append(ann)
            
        # Create filename to image_id mapping
        self.filename_to_id = {}
        for coco in self.coco_data:
            for img in coco['images']:
                self.filename_to_id[img['file_name']] = img['id']

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_data(index)

        transformed = self._augmentations(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        return image, torch.multiply(mask.type(torch.float32), 1. / 255.).permute((2, 0, 1))

    def _load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_name = self._images_list[index]

        # Check if file exists
        if not os.path.isfile(f'{image_name}'):
            raise FileNotFoundError(f"Image file not found: {image_name}")
        
        # Load image
        # print(f"siema siema, sciezka: {image_name}")
        frame = cv2.imread(f'{image_name}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Create empty mask with single channel for binary segmentation
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        
        # Get image id and its annotations
        img_id = self.filename_to_id[Path(image_name).name]
        annotations = self.image_to_masks.get(img_id, [])
        
        # Fill mask with segmentation data
        for ann in annotations:
            segmentation = ann['segmentation']
            
            # Convert COCO segmentation format to binary mask
            for seg in segmentation:
                # Convert polygon to points
                pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                # Fill the polygon
                cv2.fillPoly(mask[:, :, 0], [pts], 255)

        return frame, mask

    def __len__(self) -> int:
        return len(self._images_list)
