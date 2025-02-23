# map-segmentation-2025
based on https://github.com/PUTvision/LandCoverSeg

## Build Instructions

### Building the Project

After installing the prerequisites:

Clone the repository:
```bash
git clone https://github.com/spicyholo/map-segmentation-2025.git
cd map-segmentation-2025
```

Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate

# On Linux
source venv/bin/activate
pip install -r requirements.txt
```

Opitionally: Add CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Dataset
- trained on two types of map, we chose "Poznan 2022 aerial orthophoto high resolution", the other team annotated pictures from "true ortofotomap". 
We tested models trained on our single dataset and with the whole. 
- The overall amount of images was 400 
- There was no preprocessing implemented. We did not think any was necessary, given the complicated segmentation task. Pictures were resized.
- [dataset](https://drive.google.com/drive/folders/1NFnOefuWQ-UJp2E-DoZdNioKm9o2PYEL?usp=sharing)
- All pictures are .png, the annotations are in .json file with coco format.

## Training
There were used two models for reference.  
First was SegFormer, default parameters, encoder "efficientnet-b2"
Results:
### SegFormer, encoder: efficientnet-b2
| Metric     | Value                |
|------------|----------------------|
| test_dice  | 0.8568132519721985   |
| test_iou   | 0.7683026790618896   |
| test_loss  | 2.399618148803711    |

### DeepLabV3Plus, encoder: tu-semnasnet_100
| Test Metric      | DataLoader 0              |
|------------------|---------------------------|
| test_dice        | 0.7736                    |
| test_iou         | 0.6625                    |
| test_loss        | 0.3725                    |

### UNet, encoder: resnet50
| Test Metric      | DataLoader 0              |
|------------------|---------------------------|
| test_dice        | 0.8068                    |
| test_iou         | 0.7092                    |
| test_loss        | 0.3506                    |

### Segformer, encoder: mit_b2
| Test Metric      | DataLoader 0              |
|------------------|---------------------------|
| test_dice        | 0.8334                    |
| test_iou         | 0.7376                    |
| test_loss        | 0.3343                    |

- augmentation methods were used, already implemented inside the template
- training: python run.py name="some_name"

## Results
### For SegFormer:
![Original Image](./pictures/Seg.png)
![Original Image](./pictures/Seg_deepness.png)
### For Deeplab:
![Original Image](./pictures/deeplab.png)
![Original Image](./pictures/deeplab_deepness.png)


## Trained model in ONNX ready for `Deepness` plugin
- [model](https://drive.google.com/drive/folders/1NFnOefuWQ-UJp2E-DoZdNioKm9o2PYEL?usp=sharing), in catalog "model with metadata" are two models with configured metadata for deepness plugin for QGIS.
- deepness parameters: 10cm/px, 512px
- deepness doesnt support binary segmentation, it forces to use softmax
- exporting to onyx: python run.py name=landseg eval_mode=True ckpt_path=path export.export_to_onnx=True

## People
- Konrad Makowski, Jan Lubina

