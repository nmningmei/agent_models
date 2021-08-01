#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:33:27 2021

@author: nmei
"""
import os
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

working_dir1    = '../../another_git/agent_models/results'
working_dir2    = '../results/first_layer'
n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
def load_data(working_data,):
    df              = []
    for f in working_data:
        if ('inception' not in f):
            temp                                    = pd.read_csv(f)
            k                                       = f.split('/')[-2]
            try:
                model,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
            except:
                _model,_model_,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
                model = f'{_model}_{_model_}'
                
            if 'drop' not in temp.columns:
                temp['drop']                        = float(drop)
            df.append(temp)
    df              = pd.concat(df)
    
    
    x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
    inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
    print(x_map,inverse_x_map)
    
    df['x']         = df['noise_level'].round(9).map(x_map)
    df['x_id']      = df['noise_level'].round(9).map(x_map)
    print(df['x'].values)
    
    df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
    df                  = df.sort_values(['hidden_activation','output_activation'])
    df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']
    return df

working_data1 = glob(os.path.join(working_dir1,'*','*.csv'))
working_data2 = glob(os.path.join(working_dir2,'*','decodings.csv'))

df1 = load_data(working_data1)
df2 = load_data(working_data2)
df1 = df1[np.logical_or(df1['model_name'] == 'vgg19',
                        df1['model_name'] == 'resnet')]
df2 = df2[df2['model_name'] != 'densenet169']

name_mapper = dict(vgg19_bn='vgg19',
                   resnet50='resnet')

col_for_comparison = ['x_id',
                      'model_name',
                      'hidden_units',
                      'hidden_activation',
                      'drop',
                      'output_activation',]
iterator = tqdm(df2.groupby(col_for_comparison))
df_include = []
for attributes,df_sub in iterator:
    x_id,model_name,hidden_units,hidden_activation,drop_rate,output_activation =\
        attributes
    model_name = name_mapper[model_name]
    row_picked = np.array([df1[col_name] == value for col_name,value in zip(
                                      col_for_comparison,
                                      [x_id,
                                       model_name,
                                       hidden_units,
                                       hidden_activation,
                                       drop_rate,
                                       output_activation])]).sum(0) == len(col_for_comparison)
    row = df1[row_picked]
    row = row[row['model'] == 'CNN']
    iterator.set_description(f'size = {len(row)}')
    if len(row) == 1:
        df_sub['cnn_score'] = row['score_mean'].values
        df_sub['cnn_pval'] = row['pval'].values
        df_include.append(df_sub)
df_include = pd.concat(df_include)



































