#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:08:12 2020
@author: nmei
"""

import os
import gc
import numpy as np
import pandas as pd

import torch

from torchvision import transforms

from sklearn import metrics
from sklearn.utils import shuffle

from utils_deep import (data_loader,
                        createLossAndOptimizer,
                        behavioral_evaluate,
                        build_model,
                        hidden_activation_functions,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer,
                        train_loop,
                        validation_loop
                        )
from collections import OrderedDict
from matplotlib import pyplot as plt

# experiment control
model_dir               = '../models'
train_folder            = 'greyscaled'
valid_folder            = 'experiment_images_greyscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True #
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'resnet'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = False #
n_experiment_runs       = 20

n_noise_levels          = 50
n_keep_going            = 32

results_dir             = '../confidence_results/'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(os.path.join(results_dir,model_saving_name)):
    os.mkdir(os.path.join(results_dir,model_saving_name))

if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

model_to_train = build_model(
                pretrain_model_name,
                hidden_units,
                hidden_activation,
                hidden_dropout,
                output_units,
                )
model_to_train.to(device)
model_parameters                            = filter(lambda p: p.requires_grad, model_to_train.parameters())
params                                      = sum([np.prod(p.size()) for p in model_parameters])
print(pretrain_model_name,
#      model_to_train(next(iter(train_loader))[0]),
      f'total params = {params}')


f_name = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')

loss_func,optimizer                         = createLossAndOptimizer(model_to_train,learning_rate = lr)
if (not os.path.exists(f_name)) or (testing):
    augmentations = {
            'train':transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(45,),
            transforms.RandomVerticalFlip(p = 0.5,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'valid':transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(25,),
            transforms.RandomVerticalFlip(p = 0.5,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }

    train_loader        = data_loader(
            train_root,
            augmentations   = augmentations['train'],
            batch_size      = batch_size,
            )
    valid_loader        = data_loader(
            valid_root,
            augmentations   = augmentations['valid'],
            batch_size      = batch_size,
            )
    best_valid_loss                             = torch.from_numpy(np.array(np.inf))
    losses = []
    for idx_epoch in range(n_epochs):
        # train
        print('training ...')
        train_loss                              = train_loop(
                                                    net                 = model_to_train,
                                                    loss_func           = loss_func,
                                                    optimizer           = optimizer,
                                                    dataloader          = train_loader,
                                                    device              = device,
                                                    categorical         = categorical,
                                                    idx_epoch           = idx_epoch,
                                                    print_train         = print_train,
                                                    output_activation   = output_activation,
                                                    )
        print('validating ...')
        valid_loss,y_pred,y_true,features,labels= validation_loop(
                                                    net                 = model_to_train,
                                                    loss_func           = loss_func,
                                                    dataloader          = valid_loader,
                                                    device              = device,
                                                    categorical         = categorical,
                                                    output_activation   = output_activation,
                                                    )
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        score = metrics.roc_auc_score(y_true.detach().cpu(),y_pred.detach().cpu())
        print(f'\nepoch {idx_epoch + 1}, loss = {valid_loss:6f},score = {score:.4f}')
        if valid_loss.cpu().clone().detach().type(torch.float64) < best_valid_loss:
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(model_to_train,f_name)
        else:
            model_to_train = torch.load(f_name)
        losses.append(best_valid_loss)

        if (len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break

model_to_train = torch.load(f_name)

print('set up random seeds')
torch.manual_seed(12345)
#if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device:{device}')



print('done')
