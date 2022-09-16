import torch
device = torch.device('cuda')
print(torch.cuda.get_device_name(0))
