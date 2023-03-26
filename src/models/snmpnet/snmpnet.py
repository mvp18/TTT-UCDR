import torch
import torch.nn as nn

from ..senet import se_resnet50
from ..soft_attention import SoftAttention


class SnMpNet(nn.Module):

	def __init__(self, semantic_dim=300, pretrained='imagenet', num_tr_classes=93):
	
		super(SnMpNet, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Linear(feat_dim, semantic_dim)

		self.attention_layer = SoftAttention(input_dim=feat_dim)
		self.ratio_predictor = nn.Linear(feat_dim, num_tr_classes)

	def forward(self, x):

		features = self.base_model.features(x)
		feat_attn = self.attention_layer(features)
		
		out = self.base_model.avg_pool(feat_attn)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)
		
		feat_final = out.view(out.size(0), -1)
		mixup_ratio = self.ratio_predictor(feat_final)
		
		return mixup_ratio, feat_final


class Projector(nn.Module):

	def __init__(self, args, version=1, num_classes=4):

		super(Projector, self).__init__()

		if version==1:
			sizes = [args.semantic_emb_size] + [num_classes]
		elif version==2:
			sizes = [2048] + [num_classes]
		else:
			raise ValueError('Version not supported')

		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def reset(self):
		self.init_weights()

	def forward(self, x):
		return self.projector(x)