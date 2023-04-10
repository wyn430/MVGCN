
import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def Nor_cal_loss(pred, gold, x_global, plane_vector, local_idx, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        
        batch_size = x_global.size()[0]
        
        #normalization
        #x_global = x_global / torch.sqrt(torch.sum(x_global*x_global, dim = 1, keepdim=True))
        
        cos_sim = torch.einsum('idj,idk->ijk', x_global, x_global)
        #perpendi_dot = torch.einsum('idj,idj->ij', x_global, plane_vector)
        count = 0
        #print(cos_sim.shape)
        for i in range(batch_size):
            cos_sim[i,local_idx[i],:] = 0
            cos_sim[i,:,local_idx[i]] = 0
            
            count += (cos_sim.shape[1] - local_idx[i].sum())**2
            
        
        mean_sim = torch.abs((cos_sim.sum() / count) - 1) 
        #mean_dot = torch.abs(perpendi_dot).mean() 
        
        
    else:
        batch_size = x_global.size()[0]
             
        #x_global = x_global / torch.sqrt(torch.sum(x_global*x_global, dim = 1, keepdim=True))
        cos_sim = torch.einsum('idj,idk->ijk', x_global, x_global)
        perpendi_dot = torch.einsum('idj,idk->ijk', x_global, plane_vector)
        count = 0
        #print(cos_sim.shape)
        for i in range(batch_size):
            cos_sim[i,local_idx[i],:] = 0
            cos_sim[i,:,local_idx[i]] = 0
            
            count += (cos_sim.shape[1] - local_idx[i].sum())**2
            
        
        mean_sim = torch.abs((cos_sim.sum() / count) - 1) 
        mean_dot = torch.abs(perpendi_dot).mean() 
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss + mean_sim


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
