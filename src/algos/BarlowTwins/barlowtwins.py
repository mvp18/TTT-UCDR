import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms


class BarlowDataset(data.Dataset):
    def __init__(self, fls, transform, transform_prime):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transform = transform
        self.transform_prime = transform_prime

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        # sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        
        return x1, x2, clss

    def __len__(self):
        return len(self.fls)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def Transform_BT():

    # Imagenet standards
    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=1.0),
        Solarization(p=0.0),
        transforms.ToTensor(),
        transforms.Normalize(im_mean, im_std)
    ])
    
    transform_prime = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.1),
        Solarization(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(im_mean, im_std)
    ])

    return transform, transform_prime