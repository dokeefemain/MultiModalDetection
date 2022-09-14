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

from tensorboardX import SummaryWriter

import numpy as np
from lib import RedNet_model
from lib import RedNet_data
from lib.utils import utils
from lib.utils.utils import save_ckpt
from lib.utils.utils import print_log
import imageio
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda:0")
image_w = 640
image_h = 480
batch_size = 5
workers = 0
lr = 2e-3
lr_decay = 0.8
epoch_per_decay = 100
weight_decay = 1e-4
momentum = 0.9
epochs = 1500
start_epoch = 0
save_epoch_freq = 5
print_freq = 50
summary_dir = 'lib/models/model1/summary'
ckpt_dir = 'lib/models/model1/'
checkpoint = False

class CARLA(Dataset):
    def __init__(self, transform=None, phase_train=True):
        self.phase_train = phase_train
        self.transform = transform
        tmp = pd.read_csv("data/run1/train.csv")
        self.train_files = tmp["Name"]
        tmp = pd.read_csv("data/run1/test.csv")
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
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)
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

for epoch in range(int(start_epoch), epochs):

    scheduler.step(epoch)
    local_count = 0
    last_count = 0
    end_time = time.time()
    if epoch % save_epoch_freq == 0 and epoch != start_epoch:
        save_ckpt( ckpt_dir, model, optimizer, global_step, epoch,
                  local_count, num_train)

    for batch_idx, sample in enumerate(train_loader):
        image = sample['image'].to(device)
        depth = sample['depth'].to(device)
        target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
        optimizer.zero_grad()
        pred_scales = model(image, depth, checkpoint)
        loss = CEL_weighted(pred_scales, target_scales)
        loss.backward()
        optimizer.step()
        local_count += image.data.shape[0]
        global_step += 1
        if global_step % print_freq == 0 or global_step == 1:

            time_inter = time.time() - end_time
            count_inter = local_count - last_count
            print_log(global_step, epoch, local_count, count_inter,
                      num_train, loss, time_inter)
            end_time = time.time()

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('image', grid_image, global_step)
            grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('depth', grid_image, global_step)
            grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
            writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
            writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
            last_count = local_count

save_ckpt(ckpt_dir, model, optimizer, global_step, epochs,
          0, num_train)

print("Training completed ")