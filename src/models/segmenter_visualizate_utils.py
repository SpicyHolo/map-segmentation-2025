from neptune.types import File
import torch
import random
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import cv2
import numpy as np


def get_palette(num_classes):
    # prepare and return palette
    # 'non_permable"
    palette = [
        153,51,0,
    ]
    return palette

def visualize_segmentation_predition(logger, x_s, y_s, y_pred_s, name='test_random_mask'):
    palette = get_palette(x_s.shape[1])

    for x, y, y_pred in zip(x_s, y_s, y_pred_s):
        # Convert input image to numpy and correct format
        x = np.uint8(x.permute(1, 2, 0).cpu().numpy() * 255)
        
        # Convert prediction to numpy and correct format
        y_pred = y_pred.cpu().numpy()
        # Squeeze extra dimensions and ensure correct shape
        y_pred = y_pred.squeeze()  # Remove extra dimensions
        y_pred = (y_pred > 0.5).astype(np.uint8)  # Convert to binary mask
        
        # Create colored visualization
        result_img = np.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=np.uint8)
        result_img[y_pred > 0] = [153, 51, 0]  # Apply color from palette
        result_img = cv2.medianBlur(result_img, 5)

        # Create overlay
        frame = cv2.addWeighted(x, 0.3, result_img, 0.7, 0)
        
        # Log to Neptune
        neptune_run = logger.experiment
        neptune_run["images"].log(
            File.as_image(np.float32(frame)*1./255.),
            name=name
        )

