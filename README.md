# map-segmentation-2025
based on https://github.com/PUTvision/LandCoverSeg

## Build Instructions

### Prerequisites

Before building the project, you need to install CMake and Hatch:

#### Windows

Install CMake:
- Download the installer from [CMake official website](https://cmake.org/download/)
- Run the installer and make sure to add CMake to the system PATH
- Verify installation by opening Command Prompt and running:
```bash
cmake --version
```

Install Hatch:
```bash
pip install hatch
```

#### Linux (Ubuntu/Debian)

Install CMake:
```bash
sudo apt update
sudo apt install cmake
```

Install Hatch:
```bash
pip install hatch
```

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

Install the project with Hatch:
```bash
hatch build
```

Opitionally: Add CUDA support
```bash
pip3 install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```