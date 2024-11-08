import argparse
import os
import random
from packaging import version
from datetime import datetime
from logging import info

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

from filelock import FileLock
from torchvision import datasets, transforms

import horovod
import horovod.torch as hvd

import denoising_diffusion_pytorch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
#from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data_dir',
                    help='location of the training dataset')
parser.add_argument('--profile', type=str, default=None)

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        device,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
    ):
        super().__init__()

        from pathlib import Path
        from functools import partial

        def convert_image_to_fn(img_type, image):
          if image.mode != img_type:
              return image.convert(img_type)
          return image


        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.augment_horizontal_flip = augment_horizontal_flip

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if convert_image_to is not None else nn.Identity()

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
        ])

        self.totensor = T.Compose([
#            T.Lambda(maybe_convert_fn),
            T.ToTensor(), # CHW
        ])

        self.device = device
        self.image0 = self.raw_image(0)

    def raw_image(self, index):
        from torchvision.io import read_image

        path = self.paths[index]
        img = read_image(str(path))
        img = img.to(self.device).to(torch.float32)
        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """
        from PIL import Image
        from torchvision.io import read_image

        def read_pil(path):
            img = Image.open(path)
            img = self.totensor(img)
            return img
        path = self.paths[index]
        img = read_image(str(path))
        img = self.transform(img.to(self.device).to(torch.float32))
        return img
        """
        img = self.image0
        img = self.transform(img)
        return img


class Trainer:
    def __init__(
        self,
        size,
        rank,
        model,
        data_dir,
        *,
        train_batch_size = 16,
        augment_horizontal_flip = True,
        lr = 1e-8,
        momentum = 0.5,
        log_interval = 1000,
        profile = None,
    ):
        self.size = size
        self.rank = rank

        self.model = model.cuda()
        self.image_size = model.image_size

        self.channels = model.channels
        convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)


        self.ds = Dataset(data_dir, self.image_size, 'cuda', augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.ds, num_replicas=size, rank=rank)
        self.dl = DataLoader(self.ds, batch_size = train_batch_size, sampler=self.train_sampler)

        self.lr = lr
        self.momentum = momentum

        self.log_interval = log_interval

        self.profile = profile

        self.init_hvd()

    def init_hvd(self):
        lr_scaler = 1

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr * lr_scaler,
                              momentum=self.momentum)

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(self.optimizer,
                                             named_parameters=self.model.named_parameters(),
                                             compression=hvd.Compression.none,
                                             op=hvd.Average,
                                             gradient_predivide_factor=1.0)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def train_epoch(self, epoch):
        self.model.train()
        # Horovod: set epoch to sampler for shuffling.
        self.train_sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(self.dl):
            if self.profile is not None:
                if batch_idx > 8*3 - 1:
                    break
            data = data.cuda()
            data = self.ds.transform(data)
            self.optimizer.zero_grad()
            loss = self.model(data)
            loss.backward()
            self.optimizer.step()
            if self.rank == 0 and batch_idx % self.log_interval == 0:
                info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_sampler),
                           100. * batch_idx / len(self.dl), loss.item()))

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            if self.profile is not None:
                break

def main(args):
    hvd.init()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )


    trainer = Trainer(
        hvd.size(),
        hvd.rank(),
        diffusion,
        args.data_dir,
        train_batch_size = args.train_batch_size,
        profile = args.profile
        )

    if hvd.size() == 0:
        print('Start training: ' + datetime.now())
    trainer.train(args.epochs)

if __name__ == '__main__':
    import logging 
    logging.basicConfig(level=logging.INFO, format='{asctime} {levelname:.5} {name}: {message}', style='{')
    args = parser.parse_args()

    main(args)
