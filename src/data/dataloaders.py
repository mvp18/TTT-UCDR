import os
import numpy as np
import random
from PIL import Image, ImageOps

import torch
import torchnet as tnt
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import torchvision

from scipy.spatial.distance import cdist

_BASE_PATH = '/home/spaul/windows/TTT-UCDR/src/data'


class BaselineDataset(data.Dataset):
    def __init__(self, fls, transforms=None):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss

    def __len__(self):
        return len(self.fls)


class CuMixloader(data.Dataset):
    
    def __init__(self, fls, clss, doms, dict_domain, transforms=None):
        
        self.fls = fls
        self.clss = clss
        self.domains = doms
        self.dict_domain = dict_domain
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, self.dict_domain[sample_domain]

    def __len__(self):
        return len(self.fls)


class SAKELoader(data.Dataset):
    def __init__(self, fls, cid_mask, transforms=None):
        
        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, self.cid_mask[clss]

    def __len__(self):
        return len(self.fls)


class SAKELoader_with_domainlabel(data.Dataset):
    def __init__(self, fls, cid_mask=None, transforms=None):
        
        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
            domain_label = np.array([0])
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
            domain_label = np.array([1])
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.cid_mask is not None:
            return sample, clss, self.cid_mask[clss], domain_label
        else:
            return sample, clss, domain_label

    def __len__(self):
        return len(self.fls)


class Doodle2Search_Loader(data.Dataset):
    def __init__(self, fls_sketch, fls_image, semantic_vec, tr_classes, dict_clss, transforms=None):

        self.fls_sketch = fls_sketch
        self.fls_image = fls_image
        
        self.cls_sketch = np.array([f.split('/')[-2] for f in self.fls_sketch])
        self.cls_image = np.array([f.split('/')[-2] for f in self.fls_image])

        self.tr_classes = tr_classes
        self.dict_clss = dict_clss
        
        self.semantic_vec = semantic_vec
        # self.sim_matrix = np.exp(-np.square(cdist(self.semantic_vec, self.semantic_vec, 'euclidean'))/0.1)
        cls_euc = cdist(self.semantic_vec, self.semantic_vec, 'euclidean')
        cls_euc_scaled = cls_euc/np.expand_dims(np.max(cls_euc, axis=1), axis=1)
        self.sim_matrix = np.exp(-cls_euc_scaled)

        self.transforms = transforms
        

    def __getitem__(self, item):

        sketch = ImageOps.invert(Image.open(self.fls_sketch[item])).convert(mode='RGB')
        sketch_cls = self.cls_sketch[item]
        sketch_cls_numeric = self.dict_clss.get(sketch_cls)
        
        w2v = torch.FloatTensor(self.semantic_vec[sketch_cls_numeric, :])

        # Find negative sample
        possible_classes = self.tr_classes[self.tr_classes!=sketch_cls]
        sim = self.sim_matrix[sketch_cls_numeric, :]
        sim = np.array([sim[self.dict_clss.get(x)] for x in possible_classes])
        
        # norm = np.linalg.norm(sim, ord=1) # Similarity to probability
        # sim = sim/norm
        sim /= np.sum(sim)
        
        image_neg_cls = np.random.choice(possible_classes, 1, p=sim)[0]
        image_neg = Image.open(np.random.choice(self.fls_image[np.where(self.cls_image==image_neg_cls)[0]], 1)[0]).convert(mode='RGB')

        image_pos = Image.open(np.random.choice(self.fls_image[np.where(self.cls_image==sketch_cls)[0]], 1)[0]).convert(mode='RGB')

        if self.transforms is not None:
            sketch = self.transforms(sketch)
            image_pos = self.transforms(image_pos)
            image_neg = self.transforms(image_neg)

        return sketch, image_pos, image_neg, w2v


    def __len__(self):
        return len(self.fls_sketch)


class JigsawDataset(data.Dataset):
    def __init__(self, fls, transforms=None, jig_classes=30, bias_whole_image=0.9):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])

        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image # biases the training procedure to show the whole image more often
    
        self._image_transformer = transforms['image']
        self._augment_tile = transforms['tile']
        
        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)
        self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def get_image(self, item):
        
        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        return self._image_transformer(sample)
        
    def __getitem__(self, item):
        
        img = self.get_image(item)
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
        
        return torch.cat([self._augment_tile(img), data], 0), order, self.clss[item]

    def __len__(self):
        return len(self.fls)

    def __retrieve_permutations(self, classes):
        all_perm = np.load(os.path.join(_BASE_PATH, 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rotate_img(img, rot):
    #t_ = transforms.functional.rotate()

    if rot == 0: # 0 degrees rotation
        r_img = img
        #print('0 working')
        return img
        #return torch.from_numpy(img.copy()) #img
    elif rot == 90: # 90 degrees rotation
        #img = torch.tensor(img)
        r_img = transforms.functional.rotate(img, 90)
        #img = img.permute(1,0,2)
        #r_img = img.flip
        #r_img = torch.flipud(torch.transpose(img, (1,0,2)))
        #print('90 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    elif rot == 180: # 90 degrees rotation
        #r_img = np.fliplr(np.flipud(img))
        r_img = transforms.functional.rotate(img, 180)
        #print('180 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    elif rot == 270: # 270 degrees rotation / or -90
        #r_img = np.transpose(np.flipud(img), (1,0,2))
        r_img = transforms.functional.rotate(img, 270)
        #print('270 working')
        return r_img
        #return torch.from_numpy(r_img.copy())
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotnetLoader(object):
    
    def __init__(self, dataset, batch_size=1, transform=None, num_workers=0, shuffle=True, epoch_size=None, unsupervised=True,):
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(self.dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers
        self.transform = transform

    
    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                #print('Working')
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                
                return batch

        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size, collate_fn=_collate_fun, num_workers=self.num_workers, shuffle=self.shuffle)
        
        return data_loader
    
    def __call__(self, epoch=0):
        return self.get_iterator(epoch)
    
    def __len__(self):
        return self.epoch_size / self.batch_size