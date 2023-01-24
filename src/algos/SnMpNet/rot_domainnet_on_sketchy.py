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
from options.options_snmpnet import Options
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from data.dataloaders import RotnetLoader, BaselineDataset
from algos.Rotnet.rotnet import rotnet_ttt
from models.snmpnet.snmpnet import SnMpNet
from trainer import evaluate
from utils.logger import AverageMeter
from utils import utils


def main(args):

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	print('\nDevice:{}'.format(device))

	data_input = sketchy_extended.create_trvalte_splits(args)

	tr_classes = data_input['tr_classes']
	va_classes = data_input['va_classes']
	te_classes = data_input['te_classes']

	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

		# Image transformations
	image_transforms = {
        'pre-rotate':
		transforms.Resize((args.image_size, args.image_size)),

        'post-rotate':
        transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		]),

		'eval':
		transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		])
	}

	# Model
	model = SnMpNet(semantic_dim=args.semantic_emb_size, pretrained=None, num_tr_classes=245).cuda()

	if args.is_eccv_split:
		save_folder_name = 'eccv_split'
	else:
		save_folder_name = 'random_split'

	path_cp = os.path.join(args.checkpoint_path, 'DomainNet', 'seen-quickdraw_unseen-sketch_x_real')
	path_log = os.path.join(args.save_path, 'rotnet'+str(args.rotnet_version)+'_results', 'cross', 'Sketchy', save_folder_name)
	if not os.path.isdir(path_log):
		os.makedirs(path_log)

	data_splits_ttt = data_input['splits']['query_te']
	data_splits_ttt = np.concatenate((data_splits_ttt, data_input['splits']['gallery_te']), axis=0)

	print('Number of training samples for Rot:{}.'.format(len(data_splits_ttt)))

	data_ttt = BaselineDataset(data_splits_ttt, transforms=image_transforms['pre-rotate'])
	ttt_loader = RotnetLoader(dataset=data_ttt, batch_size=args.batch_size, transform=image_transforms['post-rotate'], 
                              num_workers=args.num_workers, shuffle=True)

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

		path_cp_ttt = os.path.join(args.save_path, 'rotnet'+str(args.rotnet_version)+'_models', 'cross', 'Sketchy', save_folder_name)

		model_ttt = rotnet_ttt(ttt_loader, model, args, device)
		model_save_name = best_model_name[:-len('.pth')] + '_rot-lrc-'+str(args.lr_clf) + '_lrb-'+str(args.lr_net) +\
						  '_bs-'+str(args.batch_size) + '_e-'+str(args.epochs)

		utils.save_checkpoint({
							'epoch':args.epochs+1, 
							'model_state_dict':model_ttt.state_dict(),
							}, directory=path_cp_ttt, save_name=model_save_name, last_chkpt='')
	
	else:
		print(f'{best_model_file} not found!')
		exit(0)
	
	data_splits = data_input['splits']
	data_te_query = BaselineDataset(data_splits['query_te'], transforms=image_transforms['eval'])
	data_te_gallery = BaselineDataset(data_splits['gallery_te'], transforms=image_transforms['eval'])

	te_loader_query = DataLoader(dataset=data_te_query, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

	te_data = evaluate(te_loader_query, te_loader_gallery, model_ttt, None, epoch, args)

	outstr="mAP@200 = %.4f, Prec@200 = %.4f, mAP@all = %.4f, Prec@100 = %.4f"%(np.mean(te_data['aps@200']), te_data['prec@200'], 
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