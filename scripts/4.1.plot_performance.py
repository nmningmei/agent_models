#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:26:03 2020
@author: nmei
InceptionNet is excluded from analysis
"""

import os
from glob import glob
from tqdm import tqdm

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib
matplotlib.pyplot.switch_backend('agg')

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

sns.set_style('whitegrid')
sns.set_context('talk',rc = {'weight' : 'bold'})

working_dir     = '../results'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','*.csv'))

df              = []
for f in working_data:
    if ('inception' not in f):
        temp                                    = pd.read_csv(f)
        k                                       = f.split('/')[-2]
        model,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
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

idxs            = np.logical_or(df['model'] == 'CNN',df['model'] == 'linear-SVM')
df_plot         = df.loc[idxs,:]

g               = sns.relplot(
                x           = 'x',
                y           = 'score_mean',
                hue         = 'model',
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'model_name',
                col_order   = ['alexnet','vgg19','mobilenet','densenet','resnet',],
                row         = 'activations',
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
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
g.savefig(os.path.join(figure_dir,'CNN_performance.jpeg'),
          dpi               = 300,
          bbox_inches       = 'tight')
g.savefig(os.path.join(figure_dir,'CNN_performance (light).jpeg'),
#          dpi = 300,
          bbox_inches       = 'tight')

# plot the decoding when CNN failed
df_chance                   = []
for attrs,df_sub in tqdm(df.groupby(['model_name',
                                     'hidden_units',
                                     'hidden_activation',
                                     'output_activation',
                                     'noise_level',
                                     'drop',
                                     ])):
    if (df_sub.shape[0] > 1):
        df_cnn = df_sub[df_sub['model'] == 'CNN']
        df_svm = df_sub[df_sub['model'] == 'linear-SVM']
        if (df_cnn['pval'].values > 0.05):# and (df_svm['pval'].values < 0.05):
            df_chance.append(df_sub)
df_chance                   = pd.concat(df_chance)

idxs                            = np.logical_or(df_chance['model'] == 'CNN',df_chance['model'] == 'linear-SVM')
df_plot                         = df_chance.copy()#loc[idxs,:]
df_plot['Decode Above Chance']  = df_plot['pval'] < 0.05
df_plot = df_plot.sort_values(['hidden_units','drop','model_name'])
df_plot['hidden_units'] = df_plot['hidden_units'].astype('category')
print(pd.unique(df_plot['hidden_units']))
k                               = len(pd.unique(df_plot['hidden_units']))

g                               = sns.relplot(
                x               = 'x',
                y               = 'score_mean',
                size            = 'drop',
                hue             = 'hidden_units',
                hue_order       = pd.unique(df_plot['hidden_units']),
                style           = 'Decode Above Chance',
                style_order     = [True, False],
                row             = 'model_name',
                row_order       = ['alexnet','vgg19','mobilenet','densenet','resnet',],
                alpha           = alpha_level,
                data            = df_plot[df_plot['model'] == 'linear-SVM'],
                palette         = sns.color_palette("bright")[:k],
                aspect          = 3,
                )
(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{row_name}'))
[ax.axhline(0.5,
            linestyle           = '--',
            color               = 'black',
            alpha               = 1.,
            lw                  = 1,
            ) for ax in g.axes.flatten()]
handles, labels                 = g.axes[0][0].get_legend_handles_labels()
g._legend.remove()
for ii,color in enumerate(sns.color_palette("bright")[:k]):
    handles[ii + 1]             = Patch(facecolor = color)
g.fig.legend(handles,
             labels,
             loc = "center right",
             borderaxespad = 0.1)
g.fig.suptitle('Linear SVM decoding the hidden layers of CNNs that failed to descriminate living vs. nonliving',
               y = 1.02)
g.savefig(os.path.join(figure_dir,'decoding_performance.jpeg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'decoding_performance (light).jpeg'),
#          dpi = 300,
          bbox_inches = 'tight')

#fig,axes = plt.subplots(figsize = (70,40),
#                        nrows = pd.unique(df['model_name']).shape[0],
#                        ncols = pd.unique(df['activations']).shape[0],
#                        sharex = True, sharey = True,)
#for ii,(ax,((model_name,activations),df_sub)) in enumerate(zip(axes.flatten(),df_plot.groupby(['model_name','activations']))):
#    k = len(pd.unique(df_sub['model']))
#    print(model_name,activations,k)
#    ax = sns.scatterplot(x = 'x',
#                         y = 'score_mean',
#                         hue = 'model',
#                         style = 'hidden_units',
#                         size = 'drop',
#                         data = df_sub,
#                         alpha = alpha_level,
#                         ax = ax,
#                         palette = sns.xkcd_palette(['black','blue'][:k]),#,'yellow','green','red']),
#                         )
#    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 1.,lw = 1)
#    _ = ax.set(ylabel = 'ROC AUC',xlabel = '',title = f'{model_name} {activations}',
#               xticks = np.arange(0,51,10),
#               )
#    if ii == len(axes.flatten()) - 1:
#        handles, labels = ax.get_legend_handles_labels()
#    ax.legend().remove()
#    xticklabels = [round(inverse_x_map[item],3) for item in np.arange(0,51,10)]
#    _ = ax.set_xticklabels(xticklabels,
#                            rotation = 45,
#                            ha = 'center')
#_ = ax.set(xlabel = 'Noise Level')
#fig.legend(handles,labels,loc="center right",borderaxespad=0.1)
#plt.subplots_adjust(right=.97)
#fig.savefig(os.path.join(figure_dir,'performance.jpeg'),
#          dpi = 300,
#          bbox_inches = 'tight')
#fig.savefig(os.path.join(figure_dir,'performance (light).jpeg'),
##          dpi = 300,
#          bbox_inches = 'tight')

#fig,axes = plt.subplots(figsize = (20,40),
#                        nrows = pd.unique(df_plot['model_name']).shape[0],
#                        sharex = True, sharey = True,)
#for ax,(model_name,df_sub) in zip(axes.flatten(),df_plot.groupby(['model_name'])):
#    temp = df_sub[df_sub['model'] == 'linear-SVM']
#    k = len(pd.unique(temp['hidden_units']))
#    all_cnn_models = df[df['model_name']==model_name].shape[0]
#    failed_cnn_models = df_sub.shape[0]
#    print(model_name,k)
#    ax = sns.scatterplot(x = 'x',
#                         y = 'score_mean',
#                         hue = 'hidden_units',
#                         size = 'drop',
#                         style = 'Decode Above Chance',
#                         style_order = [True,False],
#                         data = temp,
#                         alpha = alpha_level,
#                         ax = ax,
#                         palette = sns.xkcd_palette(np.array(['black','blue','red','green','yellow'])[:k]),
#                         )
#    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 1.,lw = 1)
#    _ = ax.set(ylabel = 'ROC AUC',
#               xlabel = '',
#               xticks = np.arange(51),
#               title = f'{model_name}, {failed_cnn_models}/{all_cnn_models}',
#               xlim = (df_chance['x'].values.min() - 1,51),
#               )
#    handles, labels = ax.get_legend_handles_labels()
#    ax.legend().remove()
#xticklabels = [round(inverse_x_map[item],3) for item in np.arange(51)]
#_ = ax.set_xticklabels(xticklabels,
#                    rotation = 45,
#                    ha = 'center')
#_ = ax.set(xlabel = 'Noise Level')
## convert the circle to irrelevant patches
#from matplotlib.patches import Patch
#handles[1] = Patch(facecolor = 'black')
#handles[2] = Patch(facecolor = 'blue',)
#handles[3] = Patch(facecolor = 'red',)
#handles[4] = Patch(facecolor = 'green',)
#fig.legend(handles,labels,loc = "center right",borderaxespad=0.1)
#plt.subplots_adjust(right=.87)
#fig.suptitle('Linear SVM decoding the hidden layers of CNNs that failed to descriminate living vs. nonliving',
#             y = 0.9)
#fig.savefig(os.path.join(figure_dir,'decoding performance.jpeg'),
#          dpi = 300,
#          bbox_inches = 'tight')
#fig.savefig(os.path.join(figure_dir,'decoding performance (light).jpeg'),
##          dpi = 300,
#          bbox_inches = 'tight')
