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
from utils import utils


class Barlow_Online:

	def __init__(self, te_query, te_gallery, model, checkpoint, img_transforms, args, device):

		self.te_query = te_query
		self.te_gallery = te_gallery
		self.model = model
		self.checkpoint = checkpoint

		self.transform, self.transform_prime = Transform_BT()
		self._eval_transform = img_transforms['eval']
		self.args = args
		self.device = device

		# normalization layer for the representations z1 and z2
		self.bn_layer = nn.BatchNorm1d(args.semantic_emb_size, affine=False).to(self.device)
		self.backbone_params = list(self.model.base_model.parameters())[:-2]
		self.classifier_params = self.model.base_model.last_linear.parameters()

		self.opt_net = optim.SGD(self.backbone_params, momentum=0, weight_decay=0, lr=self.args.lr_net)
		self.opt_clf = optim.SGD(list(self.classifier_params) + list(self.bn_layer.parameters()), momentum=self.args.momentum, 
								 weight_decay=5e-4, nesterov=bool(self.args.nesterov), lr=self.args.lr_clf)

		self.bt_loss = AverageMeter()

	def adapt_single_sample(self, img):

		if self.args.online=='std':
			# print('000')
			self.model.load_state_dict(self.checkpoint['model_state_dict'])

		self.model.train()

		img_domain = img.split('/')[-3]
		if img_domain=='sketch' or img_domain=='quickdraw':
			sample = ImageOps.invert(Image.open(img)).convert(mode='RGB')
		else:
			sample = Image.open(img).convert(mode='RGB')

		for _ in range(self.args.num_iter):
			x1 = [self.transform(sample) for _ in range(self.args.batch_size_train)]
			x2 = [self.transform_prime(sample) for _ in range(self.args.batch_size_train)]

			im1 = torch.stack(x1)
			im2 = torch.stack(x2)

			im1 = im1.float().to(self.device)
			im2 = im2.float().to(self.device)

			self.opt_net.zero_grad()
			self.opt_clf.zero_grad()

			_, im_feat1 = self.model(im1)
			_, im_feat2 = self.model(im2)

			z1 = self.model.base_model.last_linear(im_feat1)
			z2 = self.model.base_model.last_linear(im_feat2)

			# z1 = projector(z1)
			# z2 = projector(z2)

			# empirical cross-correlation matrix
			# c = torch.t(z1) @ z2
			c = torch.t(self.bn_layer(z1)) @ self.bn_layer(z2)
			c.div_(self.args.batch_size_train)

			on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
			off_diag = off_diagonal(c).pow_(2).sum()
			loss = on_diag + self.args.lambd * off_diag
			loss.backward()

			self.opt_net.step()
			self.opt_clf.step()

			self.bt_loss.update(loss.item(), im1.size(0))

	def test_single_sample(self, img):

		self.model.eval()

		img_domain = img.split('/')[-3]
		img_clss = img.split('/')[-2]

		if img_domain=='sketch' or img_domain=='quickdraw':
			sample = ImageOps.invert(Image.open(img)).convert(mode='RGB')
		else:
			sample = Image.open(img).convert(mode='RGB')

		with torch.no_grad():
			inputs = self._eval_transform(sample).unsqueeze(0).to(self.device)

			_, im_feat = self.model(inputs)
			im_em = self.model.base_model.last_linear(im_feat)

		return im_em.cpu().data.numpy(), np.expand_dims(np.array(img_clss), axis=0)

	def save_model(self, path_cp, model_save_name):

		utils.save_checkpoint({
							'iter':self.args.num_iter, 
							'model_state_dict':self.model.state_dict(),
							}, directory=path_cp, save_name=model_save_name, last_chkpt='')

	def train(self):

		# Start counting time
		start = time.time()

		np.random.shuffle(self.te_query['te'])

		print('\nQuery set size: ', len(self.te_query['te']))
		for i in range(len(self.te_query['te'])):

			self.adapt_single_sample(self.te_query['te'][i])
			sk_em, cls_sk = self.test_single_sample(self.te_query['te'][i])

			# Accumulate sketch embedding
			if i == 0:
				self.acc_sk_em = sk_em
				self.acc_cls_sk = cls_sk
			else:
				self.acc_sk_em = np.concatenate((self.acc_sk_em, sk_em), axis=0)
				self.acc_cls_sk = np.concatenate((self.acc_cls_sk, cls_sk), axis=0)

			if (i+1) % self.args.log_interval == 0:
				print('[Train] [{0}/{1}]\t'
					  'BT loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_query['te']), rn=self.bt_loss))

			# if (i+1)==50:
			# 	break

		if self.args.include_gallery:
			np.random.shuffle(self.te_gallery['te_unseen_cls'])

		print('\nGallery set size: ', len(self.te_gallery['te_unseen_cls']))
		for i in range(len(self.te_gallery['te_unseen_cls'])):

			if self.args.include_gallery:
				self.adapt_single_sample(self.te_gallery['te_unseen_cls'][i])
			
			im_em, cls_im = self.test_single_sample(self.te_gallery['te_unseen_cls'][i])

			# Accumulate image embedding
			if i == 0:
				self.acc_im_em = im_em
				self.acc_cls_im = cls_im
			else:
				self.acc_im_em = np.concatenate((self.acc_im_em, im_em), axis=0)
				self.acc_cls_im = np.concatenate((self.acc_cls_im, cls_im), axis=0)

			if (i+1) % self.args.log_interval == 0:
				print('[Train] [{0}/{1}]\t'
					  'BT loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_gallery['te_unseen_cls']), rn=self.bt_loss))

			# if (i+1)==50:
			# 	break
		
		if self.args.dataset=='DomainNet':
			print('\nSeen gallery set size: ', len(self.te_gallery['te_seen_cls']))
			for i in range(len(self.te_gallery['te_seen_cls'])):

				im_seen_em, cls_seen_im = self.test_single_sample(self.te_gallery['te_seen_cls'][i])

				# Accumulate image embedding
				if i == 0:
					acc_im_seen_em = im_seen_em
					acc_cls_seen_im = cls_seen_im
				else:
					acc_im_seen_em = np.concatenate((acc_im_seen_em, im_seen_em), axis=0)
					acc_cls_seen_im = np.concatenate((acc_cls_seen_im, cls_seen_im), axis=0)

				# if (i+1)==50:
				# 	break

			self.acc_im_em_gzs = np.concatenate((self.acc_im_em, acc_im_seen_em), axis=0)
			self.acc_cls_im_gzs = np.concatenate((self.acc_cls_im, acc_cls_seen_im), axis=0)

			print('\nSeen + Unseen Gallery Emb Dim:{}'.format(self.acc_im_em_gzs.shape))

		print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}.'.format(self.acc_sk_em.shape, self.acc_im_em.shape))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")


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