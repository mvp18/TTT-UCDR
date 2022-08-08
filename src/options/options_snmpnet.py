"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='SnMpNet for UCDR/ZS-SBIR')
        
        parser.add_argument('-root', '--root_path', default='/BS/UCDR/work/datasets/', type=str)
        parser.add_argument('-path_cp', '--checkpoint_path', default='/BS/UCDR/work/pretrained_models/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        
        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='sketch', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='quickdraw', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')
        parser.add_argument('-tv', '--trainvalid', choices=[0, 1], default=0, type=int, help='whether to include val class samples during training.\
                            1 if hyperparameter tuning done with val set')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')

        # Loss weight & reg. parameters
        parser.add_argument('-wcce', '--wcce', default=1.0, type=float, help='Weight on Distance based CCE Loss')
        parser.add_argument('-wmse', '--wmse', default=0.0, type=float, help='Weight on MSE Loss')
        parser.add_argument('-wrat', '--wratio', default=0.0, type=float, help='Weight on Soft Crossentropy Loss for mixup ratio prediction')
        parser.add_argument('-alpha', '--alpha', default=0, type=float, help='Parameter to scale weights for Class Similarity Matrix')
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-mixl', '--mixup_level', type=str, choices=['feat', 'img'], default='img', help='mixup at the image or feature level')
        parser.add_argument('-beta', '--mixup_beta', type=float, default=1, help='mixup interpolation coefficient')
        parser.add_argument('-step', '--mixup_step', type=int, default=2, help='Initial warmup steps for domain and class mixing ratios.')
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        # Barlow Twins parameters
        parser.add_argument('-path_bt', '--checkpoint_bt', default='/BS/UCDR/work/BT_models/', type=str)

        # RotNet parameters
        parser.add_argument('-path_rn', '--checkpoint_rn', default='/BS/UCDR/work/RN_models/', type=str)

        parser.add_argument('--projector', default='300-300', type=str, metavar='MLP', help='projector MLP')
        parser.add_argument('-lambd', '--lambd', type=float, default=0.0051, help='redundancy reduction loss weight')
        parser.add_argument('-lrb', '--lr_net', type=float, default=1e-5, metavar='LR', help='LR for backbone')
        parser.add_argument('-lrc', '--lr_clf', type=float, default=1e-4, metavar='LR', help='LR for semantic projector')

        self.parser = parser

    
    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()