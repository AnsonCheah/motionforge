import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("capability:", torch.cuda.get_device_capability(0))
# The real test: a kernel must actually run on sm_120, not just is_available() == True.
x = torch.zeros(1, device="cuda") + 1
print("kernel ran:", x.item())
