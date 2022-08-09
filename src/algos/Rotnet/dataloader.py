from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

from pdb import set_trace as breakpoint


def rotate_img(img, rot):
    #t_ = transforms.functional.rotate()

    if rot == 0: # 0 degrees rotation
        r_img = img
        #print('0 working')
        return img
        #return torch.from_numpy(img.copy()) #img
    elif rot == 90: # 90 degrees rotation
        #img = torch.tensor(img)
        r_img = transforms.functional.rotate(img, 90)
        #img = img.permute(1,0,2)
        #r_img = img.flip
        #r_img = torch.flipud(torch.transpose(img, (1,0,2)))
        #print('90 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    elif rot == 180: # 90 degrees rotation
        #r_img = np.fliplr(np.flipud(img))
        r_img = transforms.functional.rotate(img, 180)
        #print('180 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    elif rot == 270: # 270 degrees rotation / or -90
        #r_img = np.transpose(np.flipud(img), (1,0,2))
        r_img = transforms.functional.rotate(img, 270)
        #print('270 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotnetLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = [0.485, 0.456, 0.406]
        std_pix   = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                #print('Working')
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch

        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size