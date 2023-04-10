from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from defect_data import DefectsDataset, DefectsDataset_norm, DefectsDataset_geo, DefectsDataset_norm_geo, DefectsDataset_HGCN, DefectsDataset_HGCNnorm, DefectsDataset_HGCNnorm_geo, DefectsDataset_HGCN_geo
from model import PointNet, PointNet_norm, DGCNN, DGCNN_norm, DGCNN_geo, DGCNN_norm_geo, HGCNN, HGCNN_norm, HGCNN_geo, HGCNN_norm_geo
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, Nor_cal_loss
import sklearn.metrics as metrics
import time
from tools import generate_local_idx



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    
    

def train(args, io):
    model_root = './runs/' + args.model
    file_root = model_root + '/' + time.strftime("%Y%m%d-%H%M%S") + '/'
    try:
        os.mkdir(file_root)
    except:
        os.mkdir(model_root)
        os.mkdir(file_root)
    file_name = file_root + 'log.txt'
    outstr = 'K: %d, local_K: %d, Batch: %.6f' % (args.k,
                                                  args.kl,
                                                  args.batch_size)
    
    outstr = outstr + ' ' + args.dataset + ' ' + args.model
    with open(file_name, 'a') as f:
        f.write(outstr+'\n')
    

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
        train_loader = DataLoader(DefectsDataset(partition='train', 
                                                 num_points=args.num_points, 
                                                 root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset(partition='test', 
                                                num_points=args.num_points, 
                                                root = args.dataset,
                                                data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    elif args.model == 'pointnet_norm':        
        model = PointNet_norm(args).to(device)
        train_loader = DataLoader(DefectsDataset_norm(partition='train', 
                                                      num_points=args.num_points, 
                                                      root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_norm(partition='test', 
                                                     num_points=args.num_points, 
                                                     root = args.dataset,
                                                     data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    elif args.model == 'dgcnn':        
        model = DGCNN(args).to(device)
        train_loader = DataLoader(DefectsDataset(partition='train', 
                                                 num_points=args.num_points, 
                                                 root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset(partition='test', 
                                                num_points=args.num_points, 
                                                root = args.dataset,
                                                data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    elif args.model == 'dgcnn_norm':        
        model = DGCNN_norm(args).to(device)
        train_loader = DataLoader(DefectsDataset_norm(partition='train', 
                                                      num_points=args.num_points, 
                                                      root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_norm(partition='test', 
                                                     num_points=args.num_points, 
                                                     root = args.dataset,
                                                     data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
           
    elif args.model == 'dgcnn_geo':        
        model = DGCNN_geo(args).to(device)
        train_loader = DataLoader(DefectsDataset_geo(partition='train', 
                                                 num_points=args.num_points, 
                                                 root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_geo(partition='test', 
                                                num_points=args.num_points, 
                                                root = args.dataset,
                                                data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    elif args.model == 'dgcnn_norm_geo':        
        model = DGCNN_norm_geo(args).to(device)
        train_loader = DataLoader(DefectsDataset_norm_geo(partition='train', 
                                                      num_points=args.num_points, 
                                                      root = args.dataset, data_augmentation=True), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_norm_geo(partition='test', 
                                                     num_points=args.num_points, 
                                                     root = args.dataset,
                                                     data_augmentation=False), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        
    elif args.model == 'hgcnn':        
        model = HGCNN(args).to(device)        
        train_loader = DataLoader(DefectsDataset_HGCN(partition='train', 
                                                  num_points=args.num_points, 
                                                  root = args.dataset, data_augmentation=True),
                                  num_workers=8,batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_HGCN(partition='test', 
                                                     num_points=args.num_points, 
                                                     root = args.dataset,data_augmentation=False), 
                                 num_workers=8,batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    elif args.model == 'hgcnn_norm':
        model = HGCNN_norm(args).to(device)
        train_loader = DataLoader(DefectsDataset_HGCNnorm(partition='train', 
                                                          num_points=args.num_points, 
                                                          root = args.dataset, data_augmentation=True),
                                  num_workers=8,batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_HGCNnorm(partition='test', 
                                                         num_points=args.num_points, 
                                                         root = args.dataset,data_augmentation=False), 
                                 num_workers=8,batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        
    
    elif args.model == 'hgcnn_geo':
        model = HGCNN_geo(args).to(device)
        train_loader = DataLoader(DefectsDataset_HGCN_geo(partition='train', 
                                                              num_points=args.num_points, 
                                                              root = args.dataset, data_augmentation=True),
                                  num_workers=8,batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_HGCN_geo(partition='test', 
                                                             num_points=args.num_points, 
                                                             root = args.dataset,data_augmentation=False),
                                 num_workers=8,batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        
    
    elif args.model == 'hgcnn_norm_geo':
        model = HGCNN_norm_geo(args).to(device)
        train_loader = DataLoader(DefectsDataset_HGCNnorm_geo(partition='train', 
                                                              num_points=args.num_points, 
                                                              root = args.dataset, data_augmentation=True),
                                  num_workers=8,batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(DefectsDataset_HGCNnorm_geo(partition='test', 
                                                             num_points=args.num_points, 
                                                             root = args.dataset,data_augmentation=False),
                                 num_workers=8,batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, local_idx, label, geod in train_loader:
            #(local_idx.size())
            #local_idx = generate_local_idx(data, norm)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            #print(data.shape)
            batch_size = data.size()[0]
            opt.zero_grad()
            
            if 'pointnet' in args.model or 'dgcnn' in args.model:
                if 'geo' in args.model:
                    logits = model(data, geod)
                else:
                    logits = model(data)
   
            elif 'hgcnn' in args.model:
                if 'geo' in args.model:
                    logits = model(data, local_idx, geod)  
                else:
                    logits = model(data, local_idx)
                    
            loss = criterion(logits, label)           

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        with open(file_name, 'a') as f:
                f.write(outstr+'\n')
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, local_idx, label, geod in test_loader:
            #local_idx = generate_local_idx(data, norm)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            if 'pointnet' in args.model or 'dgcnn' in args.model:
                if 'geo' in args.model:
                    logits = model(data, geod)
                else:
                    logits = model(data)
   
            if 'hgcnn' in args.model:
                if 'geo' in args.model:
                    logits = model(data, local_idx, geod)
                else:
                    logits = model(data, local_idx)
                    
            loss = criterion(logits, label)     
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        with open(file_name, 'a') as f:
                f.write(outstr+'\n')
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '%s/model.t7' % file_root)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='hgcnn', metavar='N',
                        choices=['pointnet', 'pointnet_norm', 
                                 'dgcnn', 'dgcnn_norm', 'dgcnn_geo', 'dgcnn_norm_geo', 
                                 'hgcnn', 'hgcnn_norm', 'hgcnn_geo', 'hgcnn_norm_geo'],
                        help='Model to use, [pointnet, dgcnn, hgcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--output_C', type=int, default=4, metavar='N',
                        help='number of class')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of global nearest neighbors to use')
    parser.add_argument('--kl', type=int, default=5, metavar='N',
                        help='Num of local nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--dataset', type=str, default='../../Data/Surface_Defects_pcd_extend_2000_estnorm_noise0001', help="dataset path")
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
