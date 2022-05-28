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

def load_concat_df(working_data,n_noise_levels = 50):
    df              = []
    for f in working_data:
        temp                                    = pd.read_csv(f)
        df.append(temp)
    df              = pd.concat(df)
    # rename columns
    noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
    x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
    inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
    df['x']             = df['noise_level'].round(9).map(x_map)
    df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
    df                  = df.sort_values(['hidden_activation','output_activation'])
    df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']
    df['Model name']    = df['model_name'].map({'vgg19_bn':'VGG19','resnet50':'ResNet50',
                                                'alexnet':'AlexNet',
                                                'densenet169':'DenseNet169',
                                                'mobilenet':'MobileNetV2'})
    df['Dropout rate']  = df['dropout']
    df['# of hidden units'] = df['hidden_units']
    return df,x_map,inverse_x_map

sns.set_style('white')
sns.set_context('paper',font_scale = 2)
# from matplotlib import rc
# rc('font',weight = 'bold')
# plt.rcParams['axes.labelsize'] = 45
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 45
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['ytick.labelsize'] = 32
# plt.rcParams['xtick.labelsize'] = 32

working_dir     = '../results/trained_with_noise'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','decodings.csv'))

paper_dir = figure_dir#'/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
collect_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/all_figures'
def simpleaxes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis = 'x',direction = 'in')
    ax.tick_params(axis = 'y',direction = 'out')
idx_noise_applied = 13.25 # fit and predict by a linear regression, don't forget to apply log10



model_names     = ['AlexNet','VGG19','MobileNetV2','DenseNet169','ResNet50']
n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
print(x_map,inverse_x_map)
df,x_map,inverse_x_map = load_concat_df(working_data,n_noise_levels)

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']
df['Model name']    = df['model_name'].map({'vgg19_bn':'VGG19','resnet50':'ResNet50',
                                            'alexnet':'AlexNet',
                                            'mobilenet':'MobileNetV2',
                                            'densenet169':'DenseNet169'})
df['Dropout rate']  = df['dropout']
df['# of hidden units']  = df['hidden_units']

# plot cnn and svm on hidden layer
df_plot = pd.melt(df,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'Dropout rate',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'svm_score_mean',])
temp = pd.melt(df,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'Dropout rate',
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
                size        = 'Dropout rate',
                style       = '# of hidden units',
                col         = 'Model name',
                col_order   = model_names,
                row         = 'activations',
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 2,
                height      = 2,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
# [ax.axvline(idx_noise_applied,
#             linestyle       = '--',
#             color           = 'red',
#             alpha           = 1.,
#             lw              = 1,
#             ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise level','ROC AUC')
  .set_titles(''))

[simpleaxes(ax) for ax in g.axes.flatten()]
(g.set_axis_labels('Noise level','ROC AUC')
   .set_titles('')
  .set(ylim = (0,1.01)))
for ax_title,ax in zip(model_names,
                       g.axes[0,:]):
    ax.set(title = ax_title)
for ax_label,ax in zip(np.sort(np.unique(df['activations'])),
                       g.axes[:,0]):
    ax.annotate(ax_label.replace('_',r' $\rightarrow$ '),
                xy = (0.2,0.2),)

handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'black')
handles[2]                  = Patch(facecolor = 'blue',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             bbox_to_anchor = (1.05,0.5))
g.savefig(os.path.join(paper_dir,
                        'trained with noise performance.jpg'),
          bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure8.eps'),
#           dpi = 300,
#           bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure8.png'),
#           bbox_inches = 'tight')


# direct comparison between cnn and hidden layer
df['Decoding hidden layer > CNN'] = df['svm_score_mean'].values - df['cnn_score'].values

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding hidden layer > CNN',
                hue         = 'Model name',
                hue_order   = model_names,
                size        = 'Dropout rate',
                style       = '# of hidden units',
                row         = 'hidden_activation',
                col         = 'output_activation',
                alpha       = alpha_level,
                data        = df,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 2,
                height      = 3,
                )
[ax.axhline(0.,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.axvline(idx_noise_applied,
            linestyle       = '--',
            color           = 'red',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise level',r'$\Delta$ ROC Auc')
  .set_titles(r'{row_name} $\rightarrow$ {col_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'blue')
handles[2]                  = Patch(facecolor = 'orange',)
handles[3]                  = Patch(facecolor = 'green',)
handles[4]                  = Patch(facecolor = 'yellow',)
handles[5]                  = Patch(facecolor = 'purple',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             bbox_to_anchor = (1.05,0.5))
g.savefig(os.path.join(paper_dir,
                        'trained with noise difference between cnn and svm.jpg'),
          dpi = 100,
          bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure9.eps'),
#           dpi = 300,
#           bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure9.png'),
#           bbox_inches = 'tight')

bins = np.array([-0.5,idx_noise_applied,n_noise_levels + 1.5],dtype = 'float')
def cut_bins(x):
    if bins[0] <= x < bins[1]:
        return 'low'
    # elif bins[1] <= x < bins[2]:
    #     return 'medium'
    else:
        return 'high'

# chance level cnn
df_chance = df[df['cnn_pval' ] > 0.05]

df_plot                         = df_chance.copy()
df_plot['Decode Above Chance']  = df_plot['svm_cnn_pval'] < 0.05
df_plot = df_plot.sort_values(['# of hidden units','Dropout rate','Model name'])
df_plot['# of hidden units'] = df_plot['# of hidden units'].astype('category')
k                               = len(pd.unique(df_plot['# of hidden units']))
df_plot['x']                    = df_plot['noise_level'].round(9).map(x_map)
df_plot['x']                    = df_plot['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df_plot['i']                    = df_plot['noise_level'].round(9).map(x_map)
df_plot['groups']               = df_plot['i'].apply(cut_bins)
# compute the proportions
temp = df_plot.groupby(['Model name','Decode Above Chance']).sum().reset_index()
temp_sum = temp['i'].values.reshape(-1,2).sum(1)
temp_sum = np.repeat(temp_sum,2)
temp['sum'] = temp_sum
temp['proportion'] = temp['i'] / temp['sum']

g                               = sns.relplot(
                x               = 'x',
                y               = 'svm_score_mean',
                size            = 'Dropout rate',
                hue             = '# of hidden units',
                hue_order       = pd.unique(df_plot['# of hidden units']),
                style           = 'Decode Above Chance',
                style_order     = [True, False],
                row             = 'Model name',
                row_order       = model_names,
                alpha           = alpha_level,
                data            = df_plot,
                palette         = sns.color_palette("bright")[:k],
                height          = 5,
                aspect          = 2,
                )
(g.set_axis_labels('Noise level','ROC AUC')
  .set_titles('{row_name}')
  .set(xlim = (-0.1,50.5),ylim = (0,None)))

[ax.axhline(0.5,
            linestyle           = '--',
            color               = 'black',
            alpha               = 1.,
            lw                  = 1,
            ) for ax in g.axes.flatten()]
[ax.axvline(idx_noise_applied,
            linestyle       = '--',
            color           = 'red',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

df_proportion = []
for model_name,ax in zip(model_names,g.axes.flatten()):
    df_sub = df_plot[df_plot['Model name'] == model_name]
    temp = df_sub.groupby(['Model name','groups','Decode Above Chance']).sum().reset_index()
    temp_sum = temp['i'].values.reshape(-1,2).sum(1)
    temp_sum = np.repeat(temp_sum,2)
    temp['sum'] = temp_sum
    temp['proportion'] = temp['i'] / temp['sum']
    df_proportion.append(temp)
    
    
    tiny_ax = ax.inset_axes([.6,.15,.25,.25])
    tiny_ax = sns.barplot(x = 'groups',
                          order = ['low','high'],
                          y = 'proportion',
                          hue = 'Decode Above Chance',
                          hue_order = [True,False],
                          data = temp,
                          ax = tiny_ax,
                          palette = ['green','red'],
                          )
    # tiny_ax.set(xticklabels = ['low','medium','high'],
    tiny_ax.set_xlabel('Noise level',fontsize = 16)
    tiny_ax.set_ylabel('Decoding rate',fontsize = 16)
    tiny_ax.set(ylim = (0,1))
    tiny_handles,tiny_labels = tiny_ax.get_legend_handles_labels()
    tiny_ax.get_legend().remove()
    simpleaxes(tiny_ax)
    
df_proportion = pd.concat(df_proportion)
df_proportion.to_csv(os.path.join(paper_dir,#.replace('figures','stats'),
                                  'trained with noise CNN_chance_decode_proportion.csv'),
                      index = False)
handles, labels                 = g.axes[0][0].get_legend_handles_labels()
[handles.append(item) for item in tiny_handles]
[labels.append(item) for item in ['Decode Above Chance','Decode At Chance']]
g._legend.remove()
for ii,color in enumerate(sns.color_palette("bright")[:k]):
    handles[ii + 1]             = Patch(facecolor = color)
g.fig.legend(handles,
              labels,
              loc = "center right",
              bbox_to_anchor = (1.05,0.5))
g.savefig(os.path.join(paper_dir,
                        'trained with noise chance cnn.jpg'),
          dpi = 100,
          bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure10.eps'),
#           dpi = 300,
#           bbox_inches = 'tight')
# g.savefig(os.path.join(collect_dir,'supfigure10.png'),
#           bbox_inches = 'tight')
















