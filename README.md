# map-segmentation-2025
based on https://github.com/PUTvision/LandCoverSeg

## Build Instructions

### Prerequisites

Before building the project, you need to install CMake and Hatch:

#### Windows

1. Install CMake:
   - Download the installer from [CMake official website](https://cmake.org/download/)
   - Run the installer and make sure to add CMake to the system PATH
   - Verify installation by opening Command Prompt and running:
    ```bash
    cmake --version
    ```

2. Install Hatch:
   ```bash
   pip install hatch
   ```

#### Linux (Ubuntu/Debian)

1. Install CMake:
   ```bash
   sudo apt update
   sudo apt install cmake
   ```

2. Install Hatch:
   ```bash
   pip install hatch
   ```

### Building the Project

After installing the prerequisites:

1. Clone the repository:
   ```bash
   git clone https://github.com/spicyholo/map-segmentation-2025.git
   cd map-segmentation-2025
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Linux
   source venv/bin/activate
   ```

3. Install the project with Hatch:
   ```bash
   hatch build
   ```

4. Opitionally: Add CUDA support
   ```bash
   pip3 install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
   ```