import time
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.logger import AverageMeter


def barlow_ttt(loader, model, args):

	# normalization layer for the representations z1 and z2
	bn_layer = nn.BatchNorm1d(args.semantic_emb_size, affine=False).cuda()
	
	backbone_params = list(model.base_model.parameters())[:-2]
	classifier_params = model.base_model.last_linear.parameters()

	opt_net = optim.SGD(backbone_params, momentum=0, weight_decay=0, lr=args.lr_net)
	opt_clf = optim.SGD(list(classifier_params) + list(bn_layer.parameters()), momentum=0.9, weight_decay=5e-4, nesterov=True, lr=args.lr_clf)

	model.train()
	bt_loss = AverageMeter()

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()

		for i, (im1, im2, _) in enumerate(loader):
			
			im1 = im1.float().cuda()
			im2 = im2.float().cuda()

			opt_net.zero_grad()
			opt_clf.zero_grad()

			_, im_feat1 = model(im1)
			_, im_feat2 = model(im2)

			z1 = model.base_model.last_linear(im_feat1)
			z2 = model.base_model.last_linear(im_feat2)

			# z1 = projector(z1)
			# z2 = projector(z2)

			# empirical cross-correlation matrix
			# c = torch.t(z1) @ z2
			c = torch.t(bn_layer(z1)) @ bn_layer(z2)
			c.div_(args.batch_size)

			on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
			off_diag = off_diagonal(c).pow_(2).sum()
			loss = on_diag + args.lambd*off_diag
			loss.backward()

			opt_net.step()
			opt_clf.step()

			bt_loss.update(loss.item(), im1.size(0))

			if (i+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'BT loss: {bt.val:.4f} ({bt.avg:.4f})\t'
					  .format(epoch+1, i+1, len(loader), bt=bt_loss))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model


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