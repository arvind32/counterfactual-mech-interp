import torch
import torchvision

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)  
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(
        f"GPU {i}: {torch.cuda.get_device_name(i)}, "
        f"{props.total_memory / 1024**3:.1f} GB"
    )