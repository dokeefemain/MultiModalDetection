import os
import time

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
from PIL import Image
from tqdm import tqdm

from tensorboardX import SummaryWriter

import numpy as np
from lib import RedNet_model
from lib import RedNet_data
from lib.utils import utils
from lib.utils.utils import save_ckpt
from lib.utils.utils import print_log
import imageio
from torch.optim.lr_scheduler import LambdaLR
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
device = torch.device("cuda")
image_w = 640
image_h = 480
batch_size = 8
workers = 4
lr = 2e-3
lr_decay = 0.8
epoch_per_decay = 100
weight_decay = 1e-4
momentum = 0.9
epochs = 1500
start_epoch = 0
save_epoch_freq = 5
print_freq = 50
summary_dir = 'lib/models/model2/summary'
ckpt_dir = 'lib/models/model2/'
checkpoint = False

class CARLA(Dataset):
    def __init__(self, transform=None, phase_train=True):
        self.phase_train = phase_train
        self.transform = transform
        tmp = pd.read_csv("data/run2/train.csv")
        self.train_files = tmp["Name"]
        tmp = pd.read_csv("data/run2/test.csv")
        self.test_files = tmp["Name"]

    def __len__(self):
        if self.phase_train:
            return len(self.train_files)
        else:
            return len(self.test_files)

    def __getitem__(self, idx):
        if self.phase_train == True:
            files = self.train_files
        else:
            files = self.test_files
        label = np.load("data/run2/semantic/" + files[idx] + ".npy")
        depth = np.load("data/run2/depth/" + files[idx] + ".npy")
        image = imageio.v2.imread("data/run2/rgb/" + files[idx] + ".png", pilmode='RGB')
        sample = {'image':image, 'depth': depth, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample


train_data = CARLA(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                 RedNet_data.RandomScale((1.0, 1.4)),
                                                 RedNet_data.RandomHSV((0.9, 1.1),
                                                                       (0.9, 1.1),
                                                                       (25, 25)),
                                                 RedNet_data.RandomCrop(image_h, image_w),
                                                 RedNet_data.RandomFlip(),
                                                 RedNet_data.ToTensor(),
                                                 RedNet_data.Normalize()]),
                   phase_train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=workers, pin_memory=True, drop_last=False)
num_train = len(train_data)
model = RedNet_model.RedNet(pretrained=False)
CEL_weighted = utils.CrossEntropyLoss2d()
model.train()
model.to(device)
CEL_weighted.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=momentum, weight_decay=weight_decay)
global_step = 0
lr_decay_lambda = lambda epoch: lr_decay ** (epoch // epoch_per_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

writer = SummaryWriter(summary_dir)

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    losses = []
    device = "cuda"
    for batch_idx, sample in enumerate(loop):
        image = sample['image'].to(device)
        depth = sample['depth'].to(device)
        target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
        optimizer.zero_grad()
        pred_scales = model(image, depth, checkpoint)
        loss = loss_fn(pred_scales, target_scales)
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        

for epoch in range(int(start_epoch), epochs):
    scheduler.step(epoch)
    print("Epoch "+str(epoch))
    local_count = 0
    last_count = 0
    end_time = time.time()
    if epoch % save_epoch_freq == 0 and epoch != start_epoch:
        save_ckpt( ckpt_dir, model, optimizer, global_step, epoch,
                  local_count, num_train)
    train_fn(train_loader, model, optimizer, CEL_weighted)

save_ckpt(ckpt_dir, model, optimizer, global_step, epochs,
          0, num_train)

print("Training completed ")
