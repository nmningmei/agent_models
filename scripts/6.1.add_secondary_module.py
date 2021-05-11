#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 06:18:45 2021

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

from glob import glob
from tqdm import tqdm

from utils_deep import (hidden_activation_functions,
                        build_model,
                        createLossAndOptimizer,
                        train_and_validation,
                        data_loader,
                        validation_loop,
                        Find_Optimal_Cutoff
                        )

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

print('set up random seeds')
torch.manual_seed(12345)

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
device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrain_model_name     = 'resnet18'
hidden_units            = 2
hidden_func_name        = 'sigmoid'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0. # in this experiment, we should not have any dropouts because I cannot solve the math...
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 1000

n_noise_levels          = 50
n_keep_going            = 32
start_decoding          = False
to_round                = 9

results_dir             = '../results/RL'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.makedirs(os.path.join(model_dir,model_saving_name))

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))
f_name                  = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')

if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
print(f'device:{device}')

# initialize the model and some parameters
if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False
model_to_train = build_model(
    pretrain_model_name,
    hidden_units,
    hidden_activation,
    hidden_dropout,
    output_units,
    )
loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)

# train the model on augmented images, no noise added
model_to_train = train_and_validation(
    model_to_train,
    f_name,
    output_activation,
    loss_func,
    optimizer,
    image_resize    = image_resize,
    device          = device,
    batch_size      = batch_size,
    n_epochs        = n_epochs,
    print_train     = True,
    patience        = patience,
    train_root      = train_root,
    valid_root      = valid_root,
    )
model_to_train = model_to_train.to('cpu')

# test the model on augmented images, no noise added
transform_steps = transforms.Compose([
    transforms.Resize((image_resize,image_resize)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(45,),
    transforms.RandomVerticalFlip(p = 0.5,),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
DataLoader = data_loader(
    train_root,
    augmentations   = transform_steps,
    batch_size      = batch_size,
    )

valid_loss,y_pred,y_true,features,labels = validation_loop(
    model_to_train,
    loss_func,
    DataLoader,
    'cpu',
    categorical = categorical,
    output_activation = output_activation,)
features = torch.cat(features).detach().cpu().numpy()
y_true = torch.cat(y_true).detach().cpu().numpy()
y_pred = torch.cat(y_pred).detach().cpu().numpy()

thr = Find_Optimal_Cutoff(y_true[:,-1],y_pred[:,-1])
print(metrics.classification_report(y_true[:,-1],y_pred[:,-1]>=thr[0]))































