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

| Metric     | Value                |
|------------|----------------------|
| test_dice  | 0.8568132519721985   |
| test_iou   | 0.7683026790618896   |
| test_loss  | 2.399618148803711    |


TODO: second one 
- augmentation methods were used, already implemented inside the template
- python run.py name="some_name"

## Results
- Example images from dataset (diverse), at least 4 images
- Examples of good and bad predictions, at least 4 images
- Metrics on the test and train dataset

## Trained model in ONNX ready for `Deepness` plugin
- [model](https://drive.google.com/drive/folders/1NFnOefuWQ-UJp2E-DoZdNioKm9o2PYEL?usp=sharing)
- TODO: deepness parameters
- python run.py name=landseg eval_mode=True ckpt_path=/home/jinlobana/map-segmentation-2025/outputs/2025-01-26/22-46-11/checkpoints/last.ckpt export.export_to_onnx=True

## Demo instructions and video
- a short video of running the model in Deepness (no need for audio), preferably converted to GIF
- what ortophoto to load in QGIS and what physical place to zoom-in. E.g. Poznan 2022 zoomed-in at PUT campus
- showing the results of running the model

## People
- Konrad Makowski, Jan Lubina

## Other information
Feel free to add other information here.