import os
import torch

# Check Python version
import sys

print("Python version:", sys.version)
print("-------------------------------")

# Check PyTorch version
print("PyTorch version:", torch.__version__)
print("-------------------------------")

# Check current working directory
print("Current working directory:", os.getcwd())
print("-------------------------------")

# Check CUDA availability
print("Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
print(f"CUDA available? {cuda_available}")

if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        device_properties = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {device_name}")
        print(f"  Compute Capability: {device_properties.major}.{device_properties.minor}")
        print(f"  Total Memory: {device_properties.total_memory / (1024 ** 3):.2f} GB")
else:
    print("No CUDA devices available.")
