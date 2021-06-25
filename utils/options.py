#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
 
    # federated arguments
    parser.add_argument('--epochs', type=int, default=301, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.25, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="sgd weight decay")
    parser.add_argument('--num_shards', type=int, default=200, help="number of shards")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # noise label arguments
    parser.add_argument('--noise_type',type=str, default='symmetric', choices=['symmetric', 'pairflip', 'clean'], help='noise type of each clients')
    parser.add_argument('--noise_rate', type=float, default=0.2,  help="noise rate of each clients")
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
   
    # save arguments
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")

    # RFL arguments
    parser.add_argument('--T_pl', type=int, help = 'T_pl', default=100)
    parser.add_argument('--feature_dim', type=int, help = 'feature dimension', default=128)
    parser.add_argument('--lambda_cen', type=float, help = 'lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help = 'lambda_e', default=0.8)
    
    args = parser.parse_args()
    return args
