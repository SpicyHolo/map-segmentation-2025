from sklearn.model_selection import train_test_split

import itertools
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, List, Tuple
from glob import glob 

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

class SegmentationDataModule(LightningDataModule):
    def __init__(self,
                 data_path: Path,
                 dataset: Dataset,
                 augment: bool,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 image_mean: Tuple[float, float, float],
                 image_std: Tuple[float, float, float],
                 number_of_workers: int,
                 val_size: float = 0.1,
                 test_size: float = 0.2,
                 random_state: int = 42
                 ):
        super().__init__()

        self._data_root = Path(data_path)
        self._dataset = dataset
        self._augment = augment
        self._batch_size = batch_size
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std
        self._number_of_workers = number_of_workers
        self._val_size = val_size
        self._test_size = test_size
        self._random_state = random_state

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            A.CenterCrop(self._image_size[0], self._image_size[1], always_apply=True),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ])

        self._augmentations = A.Compose([
            # rgb augmentations
            A.RandomGamma(gamma_limit=(80, 120)),
            A.ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5),
            A.ISONoise(color_shift=(0.01, 0.1)),
            # geometry augmentations
            A.Affine(rotate=(-5, 5), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # transforms
            A.RandomCrop(self._image_size[0], self._image_size[1], always_apply=True),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None):
        # Get all image files from data directory
        all_images = [f.name for f in self._data_root.glob("*.jpg")]  # Adjust pattern if needed
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {self._data_root}")

        # First split: separate test set
        train_val_split, test_split = train_test_split(
            all_images,
            test_size=self._test_size,
            random_state=self._random_state
        )

        # Second split: separate validation set from training set
        train_split, valid_split = train_test_split(
            train_val_split,
            test_size=self._val_size/(1-self._test_size),  # Adjust for remaining data
            random_state=self._random_state
        )

        self._train_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': train_split,
                'augmentations': self._augmentations if self._augment else self._transforms,
            })

        self._valid_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': valid_split,
                'augmentations': self._transforms,
            })

        self._test_dataset: Dataset = hydra.utils.instantiate({
                '_target_': self._dataset,
                'data_root': self._data_root,
                'images_list': test_split,
                'augmentations': self._transforms,
            })

        # Print split information
        print("\nDataset Split Summary:")
        print(f"Total images: {len(all_images)}")
        print(f"Training set: {len(train_split)} images")
        print(f"Validation set: {len(valid_split)} images")
        print(f"Test set: {len(test_split)} images")

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )
