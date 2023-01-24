import time
import torch
import torch.optim as optim
import torch.nn as nn

from utils.logger import AverageMeter


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
	opt_clf = optim.SGD(list(classifier_params) + list(projector.parameters()), momentum=0.9, weight_decay=5e-4, nesterov=True, lr=args.lr_clf)
	
	rot_loss = nn.CrossEntropyLoss()

	model.train()
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