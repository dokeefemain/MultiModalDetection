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
import torchvision
import skimage.transform
from tensorboardX import SummaryWriter

import numpy as np
from lib import RedNet_model
from lib import RedNet_data
from lib.utils import utils
from lib.utils.utils import save_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc, load_ckpt
from lib.utils.utils import print_log
import imageio
from torch.optim.lr_scheduler import LambdaLR
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
device = "cuda"
device_ids = [torch.device("cuda:0")]
image_w = 640
image_h = 480
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]
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
checkpoint_path = "lib/models/model1/ckpt_epoch_410.00.pth"
checkpoint = False
parallel = True

class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float()}

class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

        return sample

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

test_transform = transforms.Compose([scaleNorm(),
                                     ToTensor(),
                                     Normalize()]),
test_data = CARLA(transform=test_transform, phase_train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)



num_train = len(train_data)
model = RedNet_model.RedNet(pretrained=False)
model = nn.DataParallel(model,device_ids=device_ids)
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

def test_fn(test_loader, model):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()

            with torch.no_grad():
                pred = model(image, depth)

            output = torch.max(pred, 1)[1]
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, label)
            intersection, union = intersectionAndUnion(output, label, 3)
            acc_meter.update(acc, pix)
            a_m, b_m = macc(output, label, 3)
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    print("Test Eval")
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
    print(mAcc.mean())
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average() * 100))

        

for epoch in range(int(start_epoch), epochs):
    scheduler.step(epoch)
    print("Epoch "+str(epoch))
    local_count = 0
    last_count = 0
    end_time = time.time()
    train_fn(train_loader, model, optimizer, CEL_weighted)
    if epoch % save_epoch_freq == 0 and epoch != start_epoch:
        model.eval()
        test_fn(test_loader,model)
        save_ckpt( ckpt_dir, model, optimizer, global_step, epoch,
                  local_count, num_train)
        model.train()


save_ckpt(ckpt_dir, model, optimizer, global_step, epochs,
          0, num_train)

print("Training completed ")
