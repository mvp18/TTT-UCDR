import sys
import os
import time
import numpy as np
import pickle
import glob
from datetime import datetime

# pytorch, torch vision
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader import RotnetLoader

sys.path.append('/home/absamant/windows/TTT-UCDR/src/')
from options.options_snmpnet import Options
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from data.dataloaders import BaselineDataset
from models.snmpnet.snmpnet import SnMpNet
from algos.SnMpNet.trainer import evaluate
from utils.logger import AverageMeter
from utils import utils


np.random.seed(0)
torch.manual_seed(0)
RG = np.random.default_rng()


def rotnet_ttt(loader, model, args):
	 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sizes = [args.semantic_emb_size] + [64, 4]

	layers = []
	for i in range(len(sizes) - 2):
		layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
		layers.append(nn.BatchNorm1d(sizes[i + 1]))
		layers.append(nn.ReLU(inplace=True))
	layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
	projector = nn.Sequential(*layers).cuda()

	classifier_params = model.base_model.last_linear.parameters()

	opt_net =  torch.optim.SGD(list(classifier_params) + list(projector.parameters()),
				lr=1e-5,
                momentum=0.9,
                nesterov=True,
                weight_decay=5e-4)

	model.train()
	rn_loss = AverageMeter() 

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()
		
		bnumber = len(loader())

		for idx, batch in enumerate(loader()):
			#tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
			#tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
			#dataX = tensors['dataX']
			#labels = tensors['labels']

			dataX, labels = batch
			dataX, labels = dataX.to(device), labels.to(device)

			opt_net.zero_grad()

			_, im_feat = model(dataX)

			pred_var = projector(model.base_model.last_linear(im_feat))
			loss_total = torch.nn.CrossEntropyLoss(pred_var, labels)

			loss_total.backward()

			opt_net.step()

			rn_loss.update(loss_total.item(), bnumber)

			if (idx+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'BT loss: {bt.val:.4f} ({bt.avg:.4f})\t'
					  .format(epoch+1, i+1, len(loader), bt=loss_total))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model

def main(args):

	#va_classes = np.load('/home/soumava/Domain-Generalized-ZS-SBIR/src/data/DomainNet/val_classes.npy').tolist()
	#te_classes = np.load('/home/soumava/Domain-Generalized-ZS-SBIR/src/data/DomainNet/test_classes.npy').tolist()
	#semantic_vec = np.load('/home/soumava/Domain-Generalized-ZS-SBIR/src/data/DomainNet/w2v_domainnet.npy', allow_pickle=True, encoding='latin1').item()
	
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	print('\nDevice:{}'.format(device))

	if args.dataset=='Sketchy':
		data_input = sketchy_extended.create_trvalte_splits(args)

	if args.dataset=='DomainNet':
		data_input = domainnet.create_trvalte_splits(args)

	if args.dataset=='TUBerlin':
		data_input = tuberlin_extended.create_trvalte_splits(args)

	tr_classes = data_input['tr_classes']
	va_classes = data_input['va_classes']
	te_classes = data_input['te_classes']

	#tr_classes = np.load('/home/soumava/Domain-Generalized-ZS-SBIR/src/data/DomainNet/train_classes.npy').tolist()
	
	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

	# Image transformations
	image_transforms = {
		'train':
		transforms.Compose([
			transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
			lambda x: np.asarray(x),
			#transforms.ToTensor(),
			#transforms.Normalize(im_mean, im_std)
		]),

		'eval':
		transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			lambda x: np.asarray(x),
			#transforms.ToTensor(),
			#transforms.Normalize(im_mean, im_std)
		]),
	}

	# Model
	model = SnMpNet(semantic_dim=args.semantic_emb_size, pretrained=None, num_tr_classes=len(tr_classes)).cuda()

	save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
	if not args.include_auxillary_domains:
		save_folder_name += '_noaux'

	if args.dataset=='Sketchy':
		if args.is_eccv_split:
			save_folder_name = 'eccv_split'
		else:
			save_folder_name = 'random_split'
	
	if args.dataset=='TUBerlin':
		save_folder_name = ''

	path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
	path_log = os.path.join('./results', args.dataset, save_folder_name)
	if not os.path.isdir(path_log):
		os.makedirs(path_log)

	## Dataloader part
	data_splits_ttt = []
	for domain in [args.seen_domain, args.holdout_domain]:
		splits_query = domainnet.trvalte_per_domain(args, domain, 0, tr_classes, va_classes, te_classes)
		data_splits_ttt += splits_query['te']
	
	splits_gallery = domainnet.trvalte_per_domain(args, args.gallery_domain, 0, tr_classes, va_classes, te_classes)
	data_splits_ttt += splits_gallery['te']

	print('Number of training samples for BT:{}.'.format(len(data_splits_ttt)))

	data_ttt = BaselineDataset(np.array(data_splits_ttt), transforms=image_transforms['eval'])
	ttt_loader = RotnetLoader(dataset=data_ttt, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

	##########

	best_model_name = os.listdir(path_cp)[0]
	print(best_model_name)
	best_model_file = os.path.join(path_cp, best_model_name)

#	if os.path.isfile(best_model_file):
		
#		print("\nLoading best model from '{}'".format(best_model_file))
		# load the best model yet
#		checkpoint = torch.load(best_model_file)
#		epoch = checkpoint['epoch']
#		best_map = checkpoint['best_map']
#		model.load_state_dict(checkpoint['model_state_dict'])
#		print("Loaded best model '{0}' (epoch {1}; mAP {2:.4f})\n".format(best_model_file, epoch, best_map))

#       path_cp_ttt = os.path.join(args.checkpoint_rn, args.dataset, save_folder_name)

		#outstr = ''

#		gzs = 0
#		splits_gallery = domainnet.trvalte_per_domain(args, args.gallery_domain, gzs, tr_classes, va_classes, te_classes)
#		data_te_gallery = BaselineDataset(np.array(splits_gallery['te']), transforms=image_transforms['eval'])
		# PyTorch test loader for gallery
#		te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=64*5, shuffle=False, 
#									   num_workers=args.num_workers, pin_memory=True)

#		for domain in [args.seen_domain, args.holdout_domain]:

#			test_head_str = 'Query:' + domain + '; Gallery:' + args.gallery_domain + '; Generalized:' + str(gzs)
#			print(test_head_str)
#			outstr += test_head_str

#			splits_query = domainnet.trvalte_per_domain(args, domain, 0, tr_classes, va_classes, te_classes)
			
			
#			data_te_query = BaselineDataset(np.array(splits_query['te']), transforms=image_transforms['eval'])			
#			data_te_comb = BaselineDataset(np.array(splits_query['te'] + splits_gallery['te']), transforms=image_transforms['eval'])

			# PyTorch test loader for query
#			te_loader_query = DataLoader(dataset=data_te_query, batch_size=64*5, shuffle=False, 
#										 num_workers=args.num_workers)

#			te_loader_ttt = DataLoader(dataset=data_te_comb, batch_size=args.batch_size, shuffle=True, 
#									   num_workers=args.num_workers)

#			model_ttt = rotnet_ttt(te_loader_ttt, model, args)

#			print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')
			# te_data = evaluate(te_loader_query, te_loader_gallery, model_ttt, None, epoch, args, 'Usual')
#			te_data = evaluate(te_loader_query, te_loader_gallery, model_ttt, None, epoch, args, 'val')
		
			# outstr+="\n\nmAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f, Time = %.6f\nmAP@200 (binary) = %.4f, "\
			# 	   "Prec@200 (binary) = %.4f, mAP@all (binary) = %.4f, Prec@100 (binary) = %.4f, Time (binary) = %.6f"\
			# 	   %(np.mean(te_data['aps@200']), te_data['prec@200'], np.mean(te_data['aps@all']), te_data['prec@100'], 
			# 		te_data['time_euc'], np.mean(te_data['aps@200_bin']), te_data['prec@200_bin'], np.mean(te_data['aps@all_bin']), 
			# 		te_data['prec@100_bin'], te_data['time_bin'])

#			outstr+="\n\nmAP@200 = %.4f, Prec@200 = %.4f"%(np.mean(te_data['aps@200']), te_data['prec@200'])

#			outstr += '\n\n'
		
#		print(outstr)
#		result_file = open(os.path.join(path_log_save, best_model_name[:-len('.pth')]+'.txt'), 'w')
#		result_file.write(outstr)
#		result_file.close()

	#	print('\nTest Results saved!')

	if os.path.isfile(best_model_file):

		print("\nLoading best model from '{}'".format(best_model_file))
		# load the best model yet
		checkpoint = torch.load(best_model_file)
		epoch = checkpoint['epoch']
		best_map = checkpoint['best_map']
		model.load_state_dict(checkpoint['model_state_dict'])
		print("Loaded best model '{0}' (epoch {1}; mAP {2:.4f})\n".format(best_model_file, epoch, best_map))

		path_cp_ttt = os.path.join(args.checkpoint_bt, args.dataset, save_folder_name)

		model_ttt = rotnet_ttt(ttt_loader, model, args)
		model_save_name = best_model_name[:-len('.pth')] + '_bt-lr-'+str(args.lr_clf)+'_bs-'+str(args.batch_size)

		utils.save_checkpoint({
							'epoch':args.epochs+1, 
							'model_state_dict':model_ttt.state_dict(),
							}, directory=path_cp_ttt, save_name=model_save_name, last_chkpt='')

	else:
		data_splits = data_input['splits']
		data_te_query = BaselineDataset(data_splits['query_te'], transforms=image_transforms['eval'])
		data_te_gallery = BaselineDataset(data_splits['gallery_te'], transforms=image_transforms['eval'])

		te_loader_query = DataLoader(dataset=data_te_query, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)

		print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

		te_data = evaluate(te_loader_query, te_loader_gallery, model_ttt, None, epoch, args)
			
		outstr+="mAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(te_data['aps@200']), te_data['prec@200'], 
				np.mean(te_data['aps@all']), te_data['prec@100'])
	
	print(outstr)
	result_file = open(os.path.join(path_log, model_save_name+'.txt'), 'w')
	result_file.write(outstr)
	result_file.close()

if __name__ == '__main__':
	# Parse options
	args = Options().parse()
	print('Parameters:\t' + str(args))
	main(args)