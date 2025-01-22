import cv2
import numpy as np
from pathlib import Path
from typing import List
from albumentations import Compose, NoOp

from src.datamodules.datasets.nonPermeableSeg_dataset import nonPermeableSegDataset

def visualize_dataset(data_root: Path, images_list: List[str]):
    """
    Visualize dataset with OpenCV windows. Use:
    - Left/Right arrows to navigate images
    - Space to toggle mask overlay
    - ESC to exit
    """
    # Initialize dataset with no augmentations
    dataset = nonPermeableSegDataset(
        data_root=data_root,
        images_list=images_list,
        augmentations=Compose([NoOp()])
    )
    
    current_idx = 0
    show_mask = True
    mask_color = (0, 255, 0)  # Green for the binary mask
    
    def update_display():
        image, mask = dataset._load_data(current_idx)
        display_img = image.copy()
        
        # Add mask overlay
        if show_mask:
            overlay = display_img.copy()
            overlay[mask[:, :, 0] > 0] = mask_color
            display_img = cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0)
        
        # Add text info
        cv2.putText(display_img, f"Image: {current_idx+1}/{len(dataset)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_img, "A/D | Space | ESC", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show mask status
        status = "ON" if show_mask else "OFF"
        color = mask_color if show_mask else (128, 128, 128)
        cv2.putText(display_img, f"Mask: {status}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Dataset Viewer", display_img)
    
    update_display()
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('d'):  # next image 
            current_idx = min(current_idx + 1, len(dataset) - 1)
            update_display()
        elif key == ord('a'):  # previous image
            current_idx = max(current_idx - 1, 0)
            update_display()
        elif key == 32:  # Space
            show_mask = not show_mask
            update_display()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    data_root = Path("data/worse")
    # Get list of image files from your data directory
    images_list = [f.name for f in data_root.glob("*.jpg")]
    
    visualize_dataset(data_root, images_list) 