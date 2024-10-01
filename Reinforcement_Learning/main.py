import torch
import numpy as np
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
print(tensor)

print(f"Tensor shape: {tensor.shape}")
print(f"Tensor data type: {tensor.dtype}")