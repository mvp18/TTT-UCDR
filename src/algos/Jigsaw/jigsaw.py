import os
import time
from PIL import Image, ImageOps
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

from models.snmpnet.snmpnet import Projector
from utils.logger import AverageMeter
from utils import utils

_BASE_PATH = '/home/spaul/windows/TTT-UCDR/src/data'


class Jigsaw_Online:

	def __init__(self, te_query, te_gallery, model, checkpoint, img_transforms, args, device, jig_classes=30, bias_whole_image=0.8):

		self.permutations = self.__retrieve_permutations(jig_classes)
		self.grid_size = 3
		self.bias_whole_image = bias_whole_image # biases the training procedure to show the whole image more often
	
		self._image_transformer = img_transforms['train']['image']
		self._augment_tile = img_transforms['train']['tile']
		self._eval_transformer = img_transforms['eval']
		
		def make_grid(x):
			return torchvision.utils.make_grid(x, self.grid_size, padding=0)
		self.returnFunc = make_grid

		self.te_query = te_query
		self.te_gallery = te_gallery
		self.model = model
		self.projector = Projector(args, args.jigsaw_version, 31).to(device)
		self.checkpoint = checkpoint
		self.args = args
		self.device = device

		self.backbone_params = list(self.model.base_model.parameters())[:-2]
		self.classifier_params = self.model.base_model.last_linear.parameters()

		self.opt_net = optim.SGD(self.backbone_params, momentum=0, weight_decay=0, lr=self.args.lr_net)
		self.opt_clf = optim.SGD(list(self.classifier_params) + list(self.projector.parameters()), momentum=self.args.momentum, 
								 weight_decay=5e-4, nesterov=bool(self.args.nesterov), lr=self.args.lr_clf)

		self.jig_loss = nn.CrossEntropyLoss()
		self.loss_meter = AverageMeter()

	def get_tile(self, img, n):
		w = float(img.size[0]) / self.grid_size
		y = int(n / self.grid_size)
		x = n % self.grid_size
		tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
		tile = self._augment_tile(tile)
		return tile
		
	def get_tiled_img(self, img):
		
		n_grids = self.grid_size ** 2
		tiles = [None] * n_grids
		for n in range(n_grids):
			tiles[n] = self.get_tile(img, n)

		order = np.random.randint(len(self.permutations) + 1) # added 1 for class 0: unsorted
		if self.bias_whole_image:
			if self.bias_whole_image > random.random():
				order = 0
		
		if order == 0:
			data = tiles
		else:
			data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
			
		data = torch.stack(data, 0)
		data = self.returnFunc(data)
		
		return [data, order]

	def __retrieve_permutations(self, classes):
		all_perm = np.load(os.path.join(_BASE_PATH, 'permutations_%d.npy' % (classes)))
		# from range [1,9] to [0,8]
		if all_perm.min() == 1:
			all_perm = all_perm - 1

		return all_perm

	def adapt_single_sample(self, img):

		if self.args.online=='std':
			# print('000')
			self.model.load_state_dict(self.checkpoint['model_state_dict'])
			self.projector.reset()

		self.model.train()
		self.projector.train()

		img_domain = img.split('/')[-3]
		if img_domain=='sketch' or img_domain=='quickdraw':
			sample = ImageOps.invert(Image.open(img)).convert(mode='RGB')
		else:
			sample = Image.open(img).convert(mode='RGB')

		for _ in range(self.args.num_iter):
			inputs = [self.get_tiled_img(self._image_transformer(sample)) for _ in range(self.args.batch_size_train)]
			# extract img and label from list of tuples
			inputs_batch, labels_batch = zip(*inputs)
			inputs_batch, labels_batch = torch.stack(inputs_batch).to(self.device), torch.LongTensor(list(labels_batch)).to(self.device)

			self.opt_net.zero_grad()
			self.opt_clf.zero_grad()

			_, im_feat = self.model(inputs_batch)
			if self.args.jigsaw_version==1:
				im_feat = self.model.base_model.last_linear(im_feat)

			pred_var = self.projector(im_feat)
			loss = self.jig_loss(pred_var, labels_batch)
			loss.backward()

			self.opt_net.step()
			self.opt_clf.step()

			self.loss_meter.update(loss.item(), inputs_batch.size(0))

	def test_single_sample(self, img):

		self.model.eval()

		img_domain = img.split('/')[-3]
		img_clss = img.split('/')[-2]

		if img_domain=='sketch' or img_domain=='quickdraw':
			sample = ImageOps.invert(Image.open(img)).convert(mode='RGB')
		else:
			sample = Image.open(img).convert(mode='RGB')

		with torch.no_grad():
			inputs = self._eval_transformer(sample).unsqueeze(0).to(self.device)

			_, im_feat = self.model(inputs)
			im_em = self.model.base_model.last_linear(im_feat)

		return im_em.cpu().data.numpy(), np.expand_dims(np.array(img_clss), axis=0)

	def save_model(self, path_cp, model_save_name):

		utils.save_checkpoint({
							'iter':self.args.num_iter, 
							'model_state_dict':self.model.state_dict(),
							'projector_state_dict':self.projector.state_dict()
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
					  'jigsaw loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_query['te']), rn=self.loss_meter))

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
					  'jigsaw loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_gallery['te_unseen_cls']), rn=self.loss_meter))

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


# Jigsaw Puzzle Loss for test time training
def jigsaw_ttt(loader, model, jig_classes, args):

	if args.jigsaw_version==1:
		jig_classifier = nn.Linear(args.semantic_emb_size, jig_classes).cuda()
		classifier_params = list(model.base_model.last_linear.parameters()) + list(jig_classifier.parameters())
	elif args.jigsaw_version==2:
		jig_classifier = nn.Linear(2048, jig_classes).cuda()
		classifier_params = list(jig_classifier.parameters())
	else:
		raise ValueError('Jigsaw version not supported')
	
	# get list of parameters in model.base_model except model.base_model.last_linear
	backbone_params = list(model.base_model.parameters())[:-2]
	jigen_loss = nn.CrossEntropyLoss()

	opt_net = optim.SGD(backbone_params, momentum=0, weight_decay=0, lr=args.lr_net)
	opt_clf = optim.SGD(classifier_params, momentum=0.9, nesterov=True, weight_decay=5e-4, lr=args.lr_clf)

	model.train()
	jig_loss = AverageMeter()

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()

		for i, (data, order, cl) in enumerate(loader):
			
			data = data.float().cuda()
			data1, data2 = torch.split(data, [3, 3], dim=1)

			order = order.long().cuda()

			opt_net.zero_grad()
			opt_clf.zero_grad()

			_, im_feat = model(data2)
			if args.jigsaw_version==1:
				im_feat = model.base_model.last_linear(im_feat)
			jig = jig_classifier(im_feat)

			loss = jigen_loss(jig, order)
			loss.backward()

			opt_net.step()
			opt_clf.step()

			jig_loss.update(loss.item(), data2.size(0))

			if (i+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'Jig loss: {jig.val:.4f} ({jig.avg:.4f})\t'
					  .format(epoch+1, i+1, len(loader), jig=jig_loss))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model