import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# If CUDA is available:
print(f"GPU Device: {torch.cuda.get_device_name(0)}")

