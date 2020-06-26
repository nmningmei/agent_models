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
                        resample_ttest,
                        resample_behavioral_estimate
                        )
from matplotlib import pyplot as plt
#plt.switch_backend('agg')

print('set up random seeds')
torch.manual_seed(12345)


# experiment control
model_dir               = '../models'
train_folder            = 'greyscaled'
valid_folder            = 'experiment_images_grayscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True #
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'vgg19'
hidden_units            = 100
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 20

n_noise_levels          = 50
n_keep_going            = 32
start_decoding          = False
to_round                = 9

results_dir             = '../results/'
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

if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])


csv_saving_name     = os.path.join(results_dir,model_saving_name,'performance_results.csv')

print(noise_levels)

for var in noise_levels:
    var = round(var,to_round)
    if True:#var not in np.array(results['noise_level']).round(to_round):
        print(f'\nworking on {var:1.1e}')
        noise_folder  = os.path.join(results_dir,model_saving_name,f'{var:1.1e}')
        if not os.path.exists(noise_folder):
            os.mkdir(noise_folder)

        # define augmentation function + noise function
        augmentations = {
                'visualize':transforms.Compose([
                transforms.Resize((image_resize,image_resize)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomRotation(25,),
                transforms.RandomVerticalFlip(p = 0.5,),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: noise_fuc(x,var)),
                ]),
                'valid':transforms.Compose([
                transforms.Resize((image_resize,image_resize)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomRotation(25,),
                transforms.RandomVerticalFlip(p = 0.5,),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: noise_fuc(x,var)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            }

        valid_loader        = data_loader(
                valid_root,
                augmentations   = augmentations['valid'],
                batch_size      = batch_size,
                # here I turn on the shuffle like it is in a real experiment
                )
        visualize_loader    = data_loader(
                valid_root,
                augmentations   = augmentations['visualize'],
                batch_size      = 2 * batch_size,
                )
        # load model architecture
        print('loading the trained model')
        model_to_train      = build_model(
                        pretrain_model_name,
                        hidden_units,
                        hidden_activation,
                        hidden_dropout,
                        output_units,
                        )
        model_to_train.to(device)
        for params in model_to_train.parameters():
            params.requires_grad = False
        
        f_name              = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')
        # load trained model
        model_to_train      = torch.load(f_name)
        loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)
        
        # evaluate the model
        y_trues,y_preds,scores,features,labels = behavioral_evaluate(
                                                        model_to_train,
                                                        n_experiment_runs,
                                                        loss_func,
                                                        valid_loader,
                                                        device,
                                                        categorical         = categorical,
                                                        output_activation   = output_activation,
                                                        image_type          = f'{var:1.1e} noise',
                                                        )
        print('evaluate behaviroal performation')
        # estimate chance level scores
        np.random.seed(12345)
        yy_trues        = torch.cat(y_trues).detach().cpu().numpy()
        yy_preds        = torch.cat(y_preds).detach().cpu().numpy()
        chance_scores   = resample_behavioral_estimate(yy_trues,yy_preds,shuffle = True)

        pval            = resample_ttest_2sample(scores,chance_scores,
                                                 match_sample_size = False,
                                                 one_tail = False,
                                                 n_permutation = int(1e5),
                                                 n_jobs = -1,
                                                 verbose = 1,
                                                 )
        asf
        # save the features and labels from the hidden layer
        decode_features = torch.cat([torch.cat(run) for run in features])
        decode_labels   = torch.cat([torch.cat(run) for run in labels])

        decode_features = decode_features.detach().cpu().numpy()
        decode_labels   = decode_labels.detach().cpu().numpy()

        if categorical:
            decode_labels = decode_labels[:,-1]
        
print('done')
