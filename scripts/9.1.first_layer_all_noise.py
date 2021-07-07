#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:08:12 2020
@author: nmei

1. get output of the first layer
2. get output of the hidden layer
3. decode from both the first layer and the hidden layer
4. test through all noise levels
"""

import os
from glob import glob
from collections import OrderedDict
import numpy as np

import torch

from torchvision import transforms

from sklearn import metrics

from joblib import Parallel,delayed

from utils_deep import (data_loader,
                        define_augmentations,
                        createLossAndOptimizer,
                        train_and_validation,
                        hidden_activation_functions,
                        behavioral_evaluate,
                        build_model,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer,
                        resample_ttest,
                        resample_behavioral_estimate,
                        simple_augmentations
                        )

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
pretrain_model_name     = 'densenet169'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 20
n_noise_levels          = 50
n_permutations          = int(1e4)

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])

if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

results_dir             = '../results/first_layer'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(os.path.join(results_dir,model_saving_name)):
    os.mkdir(os.path.join(results_dir,model_saving_name))

augmentations = define_augmentations(image_resize)

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

print('set up random seeds')
torch.manual_seed(12345)
if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

# configure the model
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

# train the model
loss_func,optimizer                         = createLossAndOptimizer(model_to_train,learning_rate = lr)
model_to_train                              = train_and_validation(
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
        patience        = 5,
        train_root      = train_root,
        valid_root      = valid_root,)

model_to_train.to('cpu')
model_to_train.eval()
for param in model_to_train.parameters():
    param.requires_grad = False

to_round = 9
csv_saving_name     = os.path.join(results_dir,model_saving_name,'performance_results.csv')
results         = dict(model_name           = [],
                       hidden_units         = [],
                       hidden_activation    = [],
                       output_activation    = [],
                       dropout              = [],
                       noise_level          = [],
                       svm_score_mean       = [],
                       svm_score_std        = [],
                       svm_pval             = [],
                       cnn_score            = [],
                       cnn_pval             = [],
                       first_score_mean     = [],
                       first_score_std      = [],
                       first_score_pval     = [],
                       )
for var in noise_levels:
    var = round(var,to_round)
    valid_loader        = data_loader(
            valid_root,
            augmentations   = simple_augmentations(image_resize,var),
            batch_size      = batch_size,
            # here I turn on the shuffle like it is in a real experiment
            )
    visualize_loader    = data_loader(
            valid_root,
            augmentations   = simple_augmentations(image_resize,var),
            batch_size      = 2 * batch_size,
            )
    loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)
    # evaluate the model
    y_trues,y_preds,features,labels = behavioral_evaluate(
                        model_to_train,
                        n_experiment_runs,
                        loss_func,
                        valid_loader,
                        device,
                        categorical = categorical,
                        output_activation = output_activation,
                        )
    behavioral_scores = metrics.roc_auc_score(y_trues,y_preds)
    def _chance(y_trues,y_preds):
        from sklearn.utils import shuffle as sk_shuffle
        _y_preds = sk_shuffle(y_preds)
        return metrics.roc_auc_score(y_trues,_y_preds)
    chance_level = Parallel(n_jobs = -1,verbose = 1)(delayed(_chance)(**{
        'y_trues':y_trues,
        'y_preds':y_preds}) for _ in range(n_permutations))
    cnn_pval = (np.sum(np.array(chance_level) >= behavioral_scores) + 1) / (n_permutations + 1)
    
    decoder = make_decoder('linear-SVM',n_jobs = 1)
    decode_features = torch.cat([torch.cat(item) for item in features]).detach().cpu().numpy()
    decode_labels   = torch.cat([torch.cat(item) for item in labels  ]).detach().cpu().numpy()
    res,_,svm_cnn_pval = decode_hidden_layer(decoder,decode_features,decode_labels[:,-1],
                              n_splits = 50,
                              test_size = 0.2,)
    svm_cnn_scores = res['test_score']
    
    # get first layer
    first_layer_func = model_to_train.features[0][0].to('cpu')
    features,labels = [],[]
    for _ in range(n_experiment_runs):
        _features,_labels = [],[]
        for batch_in, batch_lab in valid_loader:
            batch_out = first_layer_func(batch_in).view(batch_size,-1)
            _features.append(batch_out)
            _labels.append(batch_lab)
        features.append(_features)
        labels.append(_labels)
    decoder = make_decoder('linear-SVM',n_jobs = 1)
    decode_features = torch.cat([torch.cat(item) for item in features]).detach().cpu().numpy()
    decode_labels   = torch.cat([torch.cat(item) for item in labels  ]).detach().cpu().numpy()
    res,_,svm_first_pval = decode_hidden_layer(decoder,decode_features,decode_labels[:,-1],
                              n_splits = 50,
                              test_size = 0.2,)
    svm_first_scores = res['test_score']
    adsf















































