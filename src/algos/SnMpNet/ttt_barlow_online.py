import sys
import os
import time
import numpy as np
import pickle
import glob
from datetime import datetime

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('/home/spaul/windows/TTT-UCDR/src/')
from options.options_ttt_online import Options
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from algos.BarlowTwins.barlowtwins import Barlow_Online
from models.snmpnet.snmpnet import SnMpNet
from utils.metrics import compute_retrieval_metrics


def main(args):

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
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		]),

		'eval':
		transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
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
	path_log = os.path.join(args.save_path, 'bt-'+str(args.online)+'_results', args.dataset, save_folder_name)
	if not os.path.isdir(path_log):
		os.makedirs(path_log)

	splits_query = domainnet.trvalte_per_domain(args, args.holdout_domain, 0, tr_classes, va_classes, te_classes)	
	splits_gallery = domainnet.trvalte_per_domain(args, args.gallery_domain, 1, tr_classes, va_classes, te_classes)

	best_model_name = os.listdir(path_cp)[0]
	best_model_file = os.path.join(path_cp, best_model_name)

	if os.path.isfile(best_model_file):

		print("\nLoading best model from '{}'".format(best_model_file))
		# load the best model yet
		checkpoint = torch.load(best_model_file)
		epoch = checkpoint['epoch']
		best_map = checkpoint['best_map']
		model.load_state_dict(checkpoint['model_state_dict'])
		print("Loaded best model '{0}' (epoch {1}; mAP {2:.4f})\n".format(best_model_file, epoch, best_map))

		barlow_online = Barlow_Online(splits_query, splits_gallery, model, checkpoint, image_transforms, args, device)

		path_cp_ttt = os.path.join(args.save_path, 'bt-'+str(args.online)+'_models', args.dataset, save_folder_name)
		model_save_name = best_model_name[:-len('.pth')] + '_gallery-' + str(args.include_gallery) + \
						  '_lrc-'+str(args.lr_clf) + '_lrb-'+str(args.lr_net) + '_bs-'+str(args.batch_size_train) + \
						  '_mom-' + str(args.momentum) + '_e-'+str(args.num_iter)

		barlow_online.train()
		barlow_online.save_model(path_cp_ttt, model_save_name)	
	else:
		print(f'{best_model_file} not found!')
		exit(0)

	outstr = ''
	
	if args.dataset=='DomainNet':
		outstr += 'Query:' + args.holdout_domain + '; Gallery:' + args.gallery_domain + '; Generalized:0'
		
		eval_data = compute_retrieval_metrics(barlow_online.acc_sk_em, barlow_online.acc_cls_sk, barlow_online.acc_im_em, barlow_online.acc_cls_im)

		test_str="\n\nmAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(eval_data['aps@200']), eval_data['prec@200'], 
						np.mean(eval_data['aps@all']), eval_data['prec@100'])

		print(test_str)
		outstr += test_str
		outstr += '\n\n'

		outstr += 'Query:' + args.holdout_domain + '; Gallery:' + args.gallery_domain + '; Generalized:1'

		eval_data_gzs = compute_retrieval_metrics(barlow_online.acc_sk_em, barlow_online.acc_cls_sk, barlow_online.acc_im_em_gzs, barlow_online.acc_cls_im_gzs)
		test_str="\n\nmAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(eval_data_gzs['aps@200']), eval_data_gzs['prec@200'], 
						np.mean(eval_data_gzs['aps@all']), eval_data_gzs['prec@100'])

		print(test_str)
		outstr += test_str
	
	else:
		eval_data = compute_retrieval_metrics(barlow_online.acc_sk_em, barlow_online.acc_cls_sk, barlow_online.acc_im_em, barlow_online.acc_cls_im)

		test_str="\n\nmAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(eval_data['aps@200']), eval_data['prec@200'], 
						np.mean(eval_data['aps@all']), eval_data['prec@100'])
						
		print(test_str)
		outstr += test_str
	
	print(outstr)
	result_file = open(os.path.join(path_log, model_save_name+'.txt'), 'w')
	result_file.write(outstr)
	result_file.close()


if __name__ == '__main__':
	# Parse options
	args = Options().parse()
	print('Parameters:\t' + str(args))
	main(args)