import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import json
import torch
from tools import generate_local_idx
np.random.seed(24)
    

class DefectsDataset(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

       
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, False, cls, False
        
    def __len__(self):
        return len(self.datapath) 
    
    
class DefectsDataset_norm(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

       
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        

        return data_set, False, cls, False
        
    def __len__(self):
        return len(self.datapath)
    
    
    
    
class DefectsDataset_geo(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        dist_fn = fn[1][:-4] + '_geodesic_dis.npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        dist_matrix = np.load(dist_fn)['data']
        dist_matrix[dist_matrix<=0] = dist_matrix.max()
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

       
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        geod_dis = torch.from_numpy(np.multiply(dist_matrix, -1)) ## multiply -1 to the geodesic distance, because topk in pytorch is to select the top k largest value

        return point_set, False, cls, geod_dis
        
    def __len__(self):
        return len(self.datapath) 
    
class DefectsDataset_norm_geo(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        dist_fn = fn[1][:-4] + '_geodesic_dis.npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        dist_matrix = np.load(dist_fn)['data']
        dist_matrix[dist_matrix<=0] = dist_matrix.max()
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

       
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        geod_dis = torch.from_numpy(np.multiply(dist_matrix, -1)) ## multiply -1 to the geodesic distance, because topk in pytorch is to select the top k largest value

        return data_set, False, cls, geod_dis
        
    def __len__(self):
        return len(self.datapath)
    
    
    
class DefectsDataset_HGCN(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        label_fn = fn[1][:-3] + 'npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        label_set = np.load(label_fn)['data']
        
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
            
        

        local_idx = generate_local_idx(label_set)
        point_set = torch.from_numpy(point_set)
        #norm_set = torch.from_numpy(norm_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        return point_set, local_idx, cls, False 
        
    def __len__(self):
        return len(self.datapath)
    
    
    
class DefectsDataset_HGCNnorm(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        label_fn = fn[1][:-3] + 'npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        label_set = np.load(label_fn)['data']
        
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
            
        
            
        
        local_idx = generate_local_idx(label_set)
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1)
        #print(data_set.shape)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return data_set, local_idx, cls, False 
        
    def __len__(self):
        return len(self.datapath) 
    
    
class DefectsDataset_HGCN_geo(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        label_fn = fn[1][:-3] + 'npz'
        dist_fn = fn[1][:-4] + '_geodesic_dis.npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        label_set = np.load(label_fn)['data']
        dist_matrix = np.load(dist_fn)['data']
        dist_matrix[dist_matrix<=0] = dist_matrix.max()
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
            
        
            
        
        local_idx = generate_local_idx(label_set)
        point_set = torch.from_numpy(point_set)
        #print(data_set.shape)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        geod_dis = torch.from_numpy(np.multiply(dist_matrix, -1)) ## multiply -1 to the geodesic distance, because topk in pytorch is to select the top k largest value

        return point_set, local_idx, cls, geod_dis 
        
    def __len__(self):
        return len(self.datapath) 
    
    
    
    
class DefectsDataset_HGCNnorm_geo(Dataset):
    def __init__(self,
                 root,
                 num_points,
                 partition='train',
                 data_augmentation=True):
        self.npoints = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'pointset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        

        self.id2cat = {v: k for k, v in self.cat.items()}
        

        self.meta = {}
        splitfile = os.path.join(self.root, '{}_files.txt'.format(partition))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []
        
        for file in filelist:
            _,category, name = file.split('/')
            
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(os.path.join(self.root, category, name))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
              
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        label_fn = fn[1][:-3] + 'npz'
        dist_fn = fn[1][:-4] + '_geodesic_dis.npz'
        #print(self.datapath[index])
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,:3]
        norm_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)[:,6:9]
        label_set = np.load(label_fn)['data']
        dist_matrix = np.load(dist_fn)['data']
        dist_matrix[dist_matrix<=0] = dist_matrix.max()
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            norm_set[:,[0,2]] = norm_set[:,[0,2]].dot(rotation_matrix)
            #point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
            
        
            
        
        local_idx = generate_local_idx(label_set)
        point_set = torch.from_numpy(point_set)
        norm_set = torch.from_numpy(norm_set)
        data_set = torch.cat((point_set, norm_set), 1) 
        #print(data_set.shape)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        geod_dis = torch.from_numpy(np.multiply(dist_matrix, -1)) ## multiply -1 to the geodesic distance, because topk in pytorch is to select the top k largest value

        return data_set, local_idx, cls, geod_dis 
        
    def __len__(self):
        return len(self.datapath) 

