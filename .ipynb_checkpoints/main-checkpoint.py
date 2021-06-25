#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
import random

import torchvision
from torchvision import transforms
import torch

from data.cifar import CIFAR10
from model.Nets import CNN
from utils.logger import Logger
from utils.sampling import sample_iid, sample_noniid
from utils.options import args_parser
from utils.noisify import noisify_label 
from utils.train import get_local_update_objects, FedAvg
from utils.test import test_img
import time


if __name__ == '__main__':

    start = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )

    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    ##############################
    # Load dataset and split users
    ##############################
    '''
    if args.dataset == 'mnist':
        from six.moves import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset_args = dict(
            root='./data/mnist',
            download=True,
        )
        dataset_train = MNIST(
            train=True,
            transform=trans_mnist,
            noise_type="clean",
            **dataset_args,
        )
        dataset_test = MNIST(
            train=False,
            transform=transforms.ToTensor(),
            noise_type="clean",
            **dataset_args,
        )
        num_classes = 10
    '''
    if args.dataset == 'cifar':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10(
            root='./data/cifar',
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=True,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    labels = np.array(dataset_train.train_labels)
    num_imgs = len(dataset_train) // args.num_shards
    args.img_size = dataset_train[0][0].shape  # used to get model
    args.num_classes = num_classes

    # Sample users (iid / non-iid)
    if args.iid:
        dict_users = sample_iid(dataset_train, args.num_users)
    else:
        dict_users = sample_noniid(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
            num_imgs=num_imgs,
        )

    ##############################
    # Add label noise to data
    ##############################

    #dict_users[i]: client i가 가지고 있는 data의 index들을 저장
    if args.noise_type != "clean":
        for user in range(args.num_users):
            data_indices = list(copy.deepcopy(dict_users[user]))

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)

            noise_index = int(len(data_indices) * args.noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=args.noise_type)
                dataset_train.train_labels[d_idx] = noisy_label

    # for logging purposes
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)

    ##############################
    # Build model
    ##############################
    net_glob = CNN()
    net_glob = net_glob.to(args.device)
    print(net_glob)

    ##############################
    # Training
    ##############################
    logger = Logger(args)

    forget_rate_schedule = []
            
    forget_rate = args.forget_rate
    exponent = 1
    forget_rate_schedule = np.ones(args.epochs) * forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)

    print("Forget Rate Schedule")
    print(forget_rate_schedule)

    # initialize f_G
    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)
    
    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        net_glob=net_glob,
    )

    for epoch in range(args.epochs):
        local_losses = []
        local_weights = []
        f_locals = []
        args.g_epoch = epoch
        
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Local Update
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args         
            
            w, loss, f_k = local.train(copy.deepcopy(net_glob).to(args.device), copy.deepcopy(f_G).to(args.device), client_num)
            
            f_locals.append(f_k)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
        w_glob = FedAvg(local_weights)  # update global weights
        net_glob.load_state_dict(w_glob)  # copy weight to net_glob

        # Update f_G
        sim = torch.nn.CosineSimilarity(dim=1) 
        tmp = 0
        w_sum = 0
        for i in f_locals:
            sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
            w_sum += sim_weight
            tmp += sim_weight * i
        f_G = torch.div(tmp, w_sum)
        
        
        # logging purposes
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss,)
            
        print('Round {:3d}'.format(epoch))
        print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()]))

        logger.write(epoch=epoch + 1, **results)
        
    logger.close()

    print("time :", time.time() - start)