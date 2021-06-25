#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy


def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(LocalUpdateRFL(**local_update_args))

    return local_update_objects


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
            
    return w_avg


class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]
        

class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs
        
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1) 
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True)
            
    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, lambda_cen, lambda_e, new_labels):
        mse = torch.nn.MSELoss(reduce=False)
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
   
        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))
        
        if self.args.g_epoch < 100:
            lambda_cen = 0.01 * (self.args.g_epoch+1)
        
        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)
             
    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]
        
        return ind_update
        
    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        
        net.eval()
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)
        
        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)    
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1
            
        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1           
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()        
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)
                
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1
 
                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:    
                        self.pseudo_labels[idx[i]] = labels[i]
                
                # For loss calculating
                new_labels = mask[small_loss_idxs]*labels[small_loss_idxs] + (1-mask[small_loss_idxs])*self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)
                
                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, self.args.lambda_cen, self.args.lambda_e, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k         
