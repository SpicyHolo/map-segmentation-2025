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