import time
import torch
import torch.optim as optim
import torch.nn as nn

from utils.logger import AverageMeter


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