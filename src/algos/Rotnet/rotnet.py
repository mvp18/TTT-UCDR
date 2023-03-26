import time
from PIL import Image, ImageOps
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from models.snmpnet.snmpnet import Projector
from utils.logger import AverageMeter
from utils import utils


class Rotnet_Online:

	def __init__(self, te_query, te_gallery, model, checkpoint, img_transforms, args, device):

		self.te_query = te_query
		self.te_gallery = te_gallery
		self.model = model
		self.projector = Projector(args, args.rotnet_version, 4).to(device)
		self.checkpoint = checkpoint
		self.img_transforms = img_transforms
		self.args = args
		self.device = device

		self.backbone_params = list(self.model.base_model.parameters())[:-2]
		self.classifier_params = self.model.base_model.last_linear.parameters()

		self.opt_net = optim.SGD(self.backbone_params, momentum=0, weight_decay=0, lr=self.args.lr_net)
		self.opt_clf = optim.SGD(list(self.classifier_params) + list(self.projector.parameters()), momentum=self.args.momentum, 
								 weight_decay=5e-4, nesterov=bool(self.args.nesterov), lr=self.args.lr_clf)

		self.rot_loss = nn.CrossEntropyLoss()
		self.rn_loss = AverageMeter()

	# Assumes that tensor is (nchannels, height, width)
	def tensor_rot_90(self, x):
		# print('+++')
		return x.flip(2).transpose(1, 2)

	def tensor_rot_180(self, x):
		# print('---')
		return x.flip(2).flip(1)

	def tensor_rot_270(self, x):
		# print('===')
		return x.transpose(1, 2).flip(2)

	def rotate_batch(self, batch):
		
		images = []
		labels = torch.randint(4, (len(batch),), dtype=torch.long)

		for img, label in zip(batch, labels):
			if label == 1:
				img = self.tensor_rot_90(img)
			elif label == 2:
				img = self.tensor_rot_180(img)
			elif label == 3:
				img = self.tensor_rot_270(img)
			images.append(img.unsqueeze(0))
		
		return torch.cat(images), labels

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
			inputs = [self.img_transforms['train'](sample) for _ in range(self.args.batch_size_train)]
			inputs = torch.stack(inputs)

			inputs_batch, labels_batch = self.rotate_batch(inputs)
			inputs_batch, labels_batch = inputs_batch.to(self.device), labels_batch.to(self.device)

			self.opt_net.zero_grad()
			self.opt_clf.zero_grad()

			_, im_feat = self.model(inputs_batch)
			if self.args.rotnet_version==1:
				im_feat = self.model.base_model.last_linear(im_feat)

			pred_var = self.projector(im_feat)
			loss = self.rot_loss(pred_var, labels_batch)
			loss.backward()

			self.opt_net.step()
			self.opt_clf.step()

			self.rn_loss.update(loss.item(), inputs_batch.size(0))

	def test_single_sample(self, img):

		self.model.eval()

		img_domain = img.split('/')[-3]
		img_clss = img.split('/')[-2]

		if img_domain=='sketch' or img_domain=='quickdraw':
			sample = ImageOps.invert(Image.open(img)).convert(mode='RGB')
		else:
			sample = Image.open(img).convert(mode='RGB')

		with torch.no_grad():
			inputs = self.img_transforms['eval'](sample).unsqueeze(0).to(self.device)

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
					  'Rotnet loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_query['te']), rn=self.rn_loss))

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
					  'Rotnet loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(i+1, len(self.te_gallery['te_unseen_cls']), rn=self.rn_loss))

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


def rotnet_ttt(loader, model, args, device):

	if args.rotnet_version==1:
		sizes = [args.semantic_emb_size] + [4]
	elif args.rotnet_version==2:
		sizes = [2048] + [4]
	else:
		raise ValueError('Rotnet version not supported')

	layers = []
	for i in range(len(sizes) - 2):
		layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
		layers.append(nn.BatchNorm1d(sizes[i + 1]))
		layers.append(nn.ReLU(inplace=True))
	
	layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
	projector = nn.Sequential(*layers).cuda()

	backbone_params = list(model.base_model.parameters())[:-2]
	classifier_params = model.base_model.last_linear.parameters()

	opt_net = optim.SGD(backbone_params, momentum=0, weight_decay=0, lr=args.lr_net)
	opt_clf = optim.SGD(list(classifier_params) + list(projector.parameters()), momentum=args.momentum, 
						weight_decay=args.l2_reg, nesterov=bool(args.nesterov), lr=args.lr_clf)
	
	rot_loss = nn.CrossEntropyLoss()

	model.train()
	projector.train()
	rn_loss = AverageMeter()

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()

		for idx, batch in enumerate(loader()):

			dataX, labels = batch
			dataX, labels = dataX.to(device), labels.to(device)

			if opt_net is not None:
				opt_net.zero_grad()
			opt_clf.zero_grad()

			_, im_feat = model(dataX)
			if args.rotnet_version==1:
				im_feat = model.base_model.last_linear(im_feat)

			pred_var = projector(im_feat)
			loss = rot_loss(pred_var, labels)
			loss.backward()

			opt_net.step()
			opt_clf.step()

			rn_loss.update(loss.item(), dataX.size(0))

			#print(epoch+1, idx+1)
			if (idx+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'Rotnet loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(epoch+1, idx+1, len(loader()), rn=rn_loss))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model


def rotsnmp_ttt(loader, model, projector, args, device):

	backbone_params = list(model.base_model.parameters())[:-2]
	classifier_params = model.base_model.last_linear.parameters()

	opt_net = optim.SGD(backbone_params, momentum=0, weight_decay=0, lr=args.lr_net)
	opt_clf = optim.SGD(list(classifier_params) + list(projector.parameters()), momentum=0.9, weight_decay=5e-4, nesterov=True, lr=args.lr_clf)
	
	rot_loss = nn.CrossEntropyLoss()

	model.train()
	projector.train()
	rn_loss = AverageMeter()

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()

		for idx, batch in enumerate(loader()):

			dataX, labels = batch
			dataX, labels = dataX.to(device), labels.to(device)

			if opt_net is not None:
				opt_net.zero_grad()
			opt_clf.zero_grad()

			_, im_feat = model(dataX)
			if args.rotnet_version==1:
				im_feat = model.base_model.last_linear(im_feat)

			pred_var = projector(im_feat)
			loss = rot_loss(pred_var, labels)
			loss.backward()

			opt_net.step()
			opt_clf.step()

			rn_loss.update(loss.item(), dataX.size(0))

			#print(epoch+1, idx+1)
			if (idx+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'Rotnet loss: {rn.val:.4f} ({rn.avg:.4f})\t'
					  .format(epoch+1, idx+1, len(loader()), rn=rn_loss))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model, projector