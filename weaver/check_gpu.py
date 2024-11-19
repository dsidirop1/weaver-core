import torch
print(torch.cuda.is_available())  # Should return True if a GPU is accessible
print(torch.cuda.device_count())  # Shows the number of GPUs detected