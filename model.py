import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #print(feature.shape)
  
    return feature, idx


def knn_geo(x, pairwise_distance, k):
    #print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    
    return idx


def get_graph_feature_geo(x, geod_dist, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    if idx is None:
        idx = knn_geo(x, geod_dist, k=k)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #print(feature.shape)
  
    return feature, idx

def knn_HGCNN(x, k, local_idx, local = True):
    local_idx = local_idx.detach().cpu().numpy()
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    batch_size = x.size()[0]
    Min = torch.min(pairwise_distance, dim = -1)[0]
    #print(Min.size())
    if not local:
        for i in range(batch_size):
            pairwise_distance[i,:,local_idx[i]] = torch.unsqueeze(Min[i], 1)
            
    if local:
        for i in range(batch_size):
            pairwise_distance[i,:,np.invert(local_idx[i])] = torch.unsqueeze(Min[i], 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #print(idx)
    return idx


def get_graph_feature_HGCNN(x, k, local_idx, local = True, idx = None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    if idx is None:
        #print(1)
        idx = knn_HGCNN(x, k, local_idx, local = local)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    #print(x.size())

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    #print(feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #print(feature.shape)
  
    return feature, idx


def knn_HGCNN_norm(x, k, local_idx, local = True):
    local_idx = local_idx.detach().cpu().numpy()
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    batch_size = x.size()[0]
    Min = torch.min(pairwise_distance, dim = -1)[0]
    #print(Min.size())
    if not local:
        for i in range(batch_size):
            pairwise_distance[i,:,local_idx[i]] = torch.unsqueeze(Min[i], 1)
            
    if local:
        for i in range(batch_size):
            pairwise_distance[i,:,np.invert(local_idx[i])] = torch.unsqueeze(Min[i], 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #print(idx)
    return idx


def get_graph_feature_HGCNN_norm(x, k, local_idx, local = True, idx = None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #print(1)
        idx = knn_HGCNN_norm(x, k, local_idx, local = local)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    #print(feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    #print(feature.shape)
  
    return feature, idx

def knn_HGCNN_geo(x, k, local_idx, pairwise_distance, local = True):
    #the input pairwise_distance is the negative of pairwise geodesic distance
    local_idx = local_idx.detach().cpu().numpy()
    
    batch_size = x.size()[0]
    Min = torch.min(pairwise_distance, dim = -1)[0]
    #print(Min.size())
    if not local:
        for i in range(batch_size):
            pairwise_distance[i,:,local_idx[i]] = torch.unsqueeze(Min[i], 1)
            
    if local:
        for i in range(batch_size):
            pairwise_distance[i,:,np.invert(local_idx[i])] = torch.unsqueeze(Min[i], 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #print(idx)
    return idx


def get_graph_feature_HGCNN_geo(x, k, local_idx, geod_dist, local = True, idx = None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #print(1)
        idx = knn_HGCNN_geo(x, k, local_idx, geod_dist, local = local)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    #print(feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    #print(feature.shape)
  
    return feature, idx


def knn_HGCNN_norm_geo(x, k, local_idx, pairwise_distance, local = True):
    #the input pairwise_distance is the negative of pairwise geodesic distance
    local_idx = local_idx.detach().cpu().numpy()
    
    batch_size = x.size()[0]
    Min = torch.min(pairwise_distance, dim = -1)[0]
    #print(Min.size())
    if not local:
        for i in range(batch_size):
            pairwise_distance[i,:,local_idx[i]] = torch.unsqueeze(Min[i], 1)
            
    if local:
        for i in range(batch_size):
            pairwise_distance[i,:,np.invert(local_idx[i])] = torch.unsqueeze(Min[i], 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #print(idx)
    return idx


def get_graph_feature_HGCNN_norm_geo(x, k, local_idx, geod_dist, local = True, idx = None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #print(1)
        idx = knn_HGCNN_norm_geo(x, k, local_idx, geod_dist, local = local)   # (batch_size, num_points, k)
    #print('idx', idx.shape)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    #print(feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    #print(feature.shape)
  
    return feature, idx



class PointNet(nn.Module):
    def __init__(self, args, output_channels=4):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    
class PointNet_norm(nn.Module):
    def __init__(self, args, output_channels=4):
        super(PointNet_norm, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        #print(batch_size)
        x, idx = get_graph_feature(x, k=self.k, idx = None)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature(x1, k=self.k, idx = None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature(x2, k=self.k, idx = None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature(x3, k=self.k, idx = None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print(x.shape)

        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    

    
class DGCNN_norm(nn.Module):
    def __init__(self, args):
        super(DGCNN_norm, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        #print(batch_size)
        x, idx = get_graph_feature(x, k=self.k, idx = None)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature(x1, k=self.k, idx = None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature(x2, k=self.k, idx = None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature(x3, k=self.k, idx = None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print(x.shape)

        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    
    
class DGCNN_geo(nn.Module):
    def __init__(self, args):
        super(DGCNN_geo, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, geod_dist):
        batch_size = x.size(0)
        #print(batch_size)
        x, idx = get_graph_feature_geo(x, geod_dist, k=self.k, idx = None)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_geo(x1, geod_dist, k=self.k, idx = None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_geo(x2, geod_dist, k=self.k, idx = None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_geo(x3, geod_dist, k=self.k, idx = None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print(x.shape)

        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    

    
class DGCNN_norm_geo(nn.Module):
    def __init__(self, args):
        super(DGCNN_norm_geo, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, geod_dist):
        batch_size = x.size(0)
        #print(batch_size)
        x, idx = get_graph_feature_geo(x, geod_dist, k=self.k, idx = None)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_geo(x1, geod_dist, k=self.k, idx = None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_geo(x2, geod_dist, k=self.k, idx = None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_geo(x3, geod_dist, k=self.k, idx = None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print(x.shape)

        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    
    
class HGCNN(nn.Module):
    def __init__(self, args):
        super(HGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
                
        self.bn9 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx):
        
        batch_size = x.size(0)
        
        #branch for local feature
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN(x_local, self.kl, local_idx, local = True, idx = None)
        #print(x_local.shape, '0')
        #print(idx.shape)
        x_local = self.conv1(x_local)
        #print(x_local.shape, '1')
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        #print(x_local_1.shape, '1')
        x_local, idx = get_graph_feature_HGCNN(x_local_1, self.kl, local_idx, local = True, idx = None)
        #print(x_local.shape, '2')
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN(x_local_2, self.kl, local_idx, local = True, idx = None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN(x_local_3, self.kl, local_idx, local = True, idx = None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]

        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)      
       
        
        #branch for global feature
        x, idx = get_graph_feature_HGCNN(x, self.k, local_idx, local = False, idx = None)
        #print(idx.size())
        x = self.conv5(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local = False, idx = None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local = False, idx = None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local = False, idx = None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        for b in range(batch_size):
            x[b, :, local_idx[b]] = x_local[b, :, local_idx[b]]
        
        #print(x.shape)
        x = self.conv9(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    
class HGCNN_norm(nn.Module):
    def __init__(self, args):
        super(HGCNN_norm, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
                
        self.bn9 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx):
        
        batch_size = x.size(0)
        #print(x.shape)
        #branch for local feature
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN_norm(x_local, self.kl, local_idx, local = True, idx = None)
        #print(x_local.shape, '0')
        
        x_local = self.conv1(x_local)
        #print(x_local.shape, '1')
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        
        x_local, idx = get_graph_feature_HGCNN_norm(x_local_1, self.kl, local_idx, local = True, idx = None)
        
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_norm(x_local_2, self.kl, local_idx, local = True, idx = None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_norm(x_local_3, self.kl, local_idx, local = True, idx = None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]

        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)      
       
        
        #branch for global feature
        x, idx = get_graph_feature_HGCNN_norm(x, self.k, local_idx, local = False, idx = None)
        #print(idx.size())
        x = self.conv5(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_HGCNN_norm(x1, self.k, local_idx, local = False, idx = None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_norm(x2, self.k, local_idx, local = False, idx = None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_norm(x3, self.k, local_idx, local = False, idx = None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        for b in range(batch_size):
            x[b, :, local_idx[b]] = x_local[b, :, local_idx[b]]
        
        #print(x.shape)
        x = self.conv9(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
class HGCNN_geo(nn.Module):
    def __init__(self, args):
        super(HGCNN_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
                
        self.bn9 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dist):
        
        batch_size = x.size(0)
        
        #branch for local feature
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN_geo(x_local, self.kl, local_idx, geod_dist, local = True, idx = None)
        #print(x_local.shape, '0')
        #print(idx.shape)
        x_local = self.conv1(x_local)
        #print(x_local.shape, '1')
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        #print(x_local_1.shape, '1')
        x_local, idx = get_graph_feature_HGCNN_geo(x_local_1, self.kl, local_idx, geod_dist, local = True, idx = None)
        #print(x_local.shape, '2')
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_geo(x_local_2, self.kl, local_idx, geod_dist, local = True, idx = None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_geo(x_local_3, self.kl, local_idx, geod_dist, local = True, idx = None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]

        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)      
       
        
        #branch for global feature
        x, idx = get_graph_feature_HGCNN_geo(x, self.k, local_idx, geod_dist, local = False, idx = None)
        #print(idx.size())
        x = self.conv5(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_HGCNN_geo(x1, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_geo(x2, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_geo(x3, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        for b in range(batch_size):
            x[b, :, local_idx[b]] = x_local[b, :, local_idx[b]]
        
        #print(x.shape)
        x = self.conv9(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    
class HGCNN_norm_geo(nn.Module):
    def __init__(self, args):
        super(HGCNN_norm_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        
        output_channels = args.output_C
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
                
        self.bn9 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dist):
        
        batch_size = x.size(0)
        #print(x.shape)
        #branch for local feature
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN_norm_geo(x_local, self.kl, local_idx, geod_dist, local = True, idx = None)
        #print(x_local.shape, '0')
        
        x_local = self.conv1(x_local)
        #print(x_local.shape, '1')
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        
        x_local, idx = get_graph_feature_HGCNN_norm_geo(x_local_1, self.kl, local_idx, geod_dist, local = True, idx = None)
        
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_norm_geo(x_local_2, self.kl, local_idx, geod_dist, local = True, idx = None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]

        x_local, idx = get_graph_feature_HGCNN_norm_geo(x_local_3, self.kl, local_idx, geod_dist, local = True, idx = None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]

        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)      
       
        
        #branch for global feature
        x, idx = get_graph_feature_HGCNN_norm_geo(x, self.k, local_idx, geod_dist, local = False, idx = None)
        #print(idx.size())
        x = self.conv5(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        

        x, idx = get_graph_feature_HGCNN_norm_geo(x1, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_norm_geo(x2, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature_HGCNN_norm_geo(x3, self.k, local_idx, geod_dist, local = False, idx = None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        for b in range(batch_size):
            x[b, :, local_idx[b]] = x_local[b, :, local_idx[b]]
        
        #print(x.shape)
        x = self.conv9(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    

    
