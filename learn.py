import torch
import torch.nn as nn

input_data = torch.rand(16,512,80)
conv = nn.Conv1d(512,1024,1)
output_data = conv(input_data)
print(output_data.shape)