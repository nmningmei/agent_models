#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:51:57 2021

@author: nmei
"""
import os
from glob import glob
from tqdm import tqdm

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib
# matplotlib.pyplot.switch_backend('agg')

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

working_dir     = '../results/first_layer'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','decodings.csv'))

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'


df              = []
for f in working_data:
    if ('densenet' not in f):
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

n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
print(x_map,inverse_x_map)

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']

# plot cnn and svm on hidden layer
df_plot = pd.melt(df,id_vars = ['model_name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'svm_score_mean',])
temp = pd.melt(df,id_vars = ['model_name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_pval',
                                'svm_cnn_pval',])
df_plot['pvals'] = temp['value'].values.copy()
df_plot['Type'] = df_plot['variable'].apply(lambda x: x.split('_')[0].upper())
df_plot['Type'] = df_plot['Type'].map({'CNN':'CNN',
                                       'SVM':'Decode hidden layer'})

g               = sns.relplot(
                x           = 'x',
                y           = 'value',
                hue         = 'Type',
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'model_name',
                col_order   = ['vgg19_bn','resnet50',],
                row         = 'activations',
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{col_name} {row_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'black')
handles[2]                  = Patch(facecolor = 'blue',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             borderaxespad  = 0.1)
g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')


# plot cnn and svm on first layer
df_plot = pd.melt(df,id_vars = ['model_name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'first_score_mean',])
temp = pd.melt(df,id_vars = ['model_name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_pval',
                                'svm_first_pval',])
df_plot['pvals'] = temp['value'].values.copy()
df_plot['Type'] = df_plot['variable'].apply(lambda x: x.split('_')[0].upper())
df_plot['Type'] = df_plot['Type'].map({'CNN':'CNN',
                                       'FIRST':'Decode first layer'})

g               = sns.relplot(
                x           = 'x',
                y           = 'value',
                hue         = 'Type',
                # size        = 'drop',
                # style       = 'hidden_units',
                col         = 'model_name',
                col_order   = ['vgg19_bn','resnet50',],
                # row         = 'activations',
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{col_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[0]                  = Patch(facecolor = 'black')
handles[1]                  = Patch(facecolor = 'blue',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             borderaxespad  = 0.1)
g.savefig(os.path.join(paper_dir,'supplymental cnn first layer decoding vgg+resnet.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplymental cnn first layer decoding vgg+resnet (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')

# direct comparison between cnn and hidden layer
df['Decoding hidden layer > CNN'] = df['svm_score_mean'].values - df['cnn_score'].values
# direct comparison between cnn and first layer
df['Decoding first layer > CNN'] = df['first_score_mean'].values - df['cnn_score'].values

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding hidden layer > CNN',
                hue         = 'model_name',
                hue_order   = ['vgg19_bn','resnet50',],
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'hidden_activation',
                row         = 'output_activation',
                alpha       = alpha_level,
                data        = df,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','Difference')
  .set_titles('{row_name} {col_name}'))
g.savefig(os.path.join(paper_dir,'hidden better than cnn.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'hidden better than cnn (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding first layer > CNN',
                hue         = 'model_name',
                hue_order   = ['vgg19_bn','resnet50',],
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'hidden_activation',
                row         = 'output_activation',
                alpha       = alpha_level,
                data        = df,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','Difference')
  .set_titles('{row_name} {col_name}'))
g.savefig(os.path.join(paper_dir,'first better than cnn.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'first better than cnn (light).jpg'),
          # dpi = 300,first
          bbox_inches = 'tight')






















