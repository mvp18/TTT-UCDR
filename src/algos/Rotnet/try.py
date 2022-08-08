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
#from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader import DataLoader

sys.path.append('/home/absamant/UCDR/src/')
from options.options_snmpnet_ssl import Options
from data.DomainNet import domainnet
from data.dataloaders import BaselineDataset
from models.deepall.deepall_seresnet50 import SnMpNet_SSL
from utils.logger import AverageMeter
#from trainer import evaluate

print('haggu')