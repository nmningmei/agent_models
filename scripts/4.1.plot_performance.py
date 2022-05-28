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

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

sns.set_style('white')
sns.set_context('paper',font_scale=2)

def simpleaxes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis = 'x',direction = 'in')
    ax.tick_params(axis = 'y',direction = 'out')

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


_,bins          = pd.cut(np.arange(50),2,retbins = True)
def cut_bins(x):
    if bins[0] <= x < bins[1]:
        return 'low'
    # elif bins[1] <= x < bins[2]:
    #     return 'medium'
    else:
        return 'high'
if __name__ == "__main__":

    working_dir     = '../results/all_for_all'#'../../another_git/agent_models/results'
    figure_dir      = '../figures'
    collect_dir     = '/export/home/nmei/nmei/properties_of_unconscious_processing/all_figures'
    working_data    = glob(os.path.join(working_dir,'*','*.csv'))
    paper_dir       = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
    
    marker_factor   = 10
    marker_type     = ['8','s','p','*','+','D','o']
    alpha_level     = .75
    model_names     = ['AlexNet','VGG19','MobileNetV2','DenseNet169','ResNet50']
    n_noise_levels  = 50

    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    _,bins          = pd.cut(np.arange(n_noise_levels),2,retbins = True)
    noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
    df,x_map,inverse_x_map = load_concat_df(working_data,n_noise_levels)
    
    
    id_vars = ['x',
               'noise_level',
               '# of hidden units','Dropout rate',
               'activations',
               'Model name',
               ]
    value_vars = ['svm_score_mean','cnn_score_mean']
    df_plot = pd.melt(df,id_vars = id_vars,value_vars = value_vars)
    df_plot['model'] = df_plot['variable'].apply(lambda x:x.split('_')[0].upper())
    df_plot['score_mean'] = df_plot['value']
    
    g               = sns.relplot(
                    x           = 'x',
                    y           = 'score_mean',
                    hue         = 'model',
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
    [ax.set(xticks = [0,n_noise_levels],
            xticklabels = [0,noise_levels.max()]
            ) for ax in g.axes.flatten()]
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
                    xy = (0.2,0.25),)
    handles, labels             = g.axes[0][0].get_legend_handles_labels()
    # convert the circle to irrelevant patches
    handles[1]                  = Patch(facecolor = 'black')
    handles[2]                  = Patch(facecolor = 'blue',)
    g._legend.remove()
    g.fig.legend(handles,
                 labels,
                 loc            = 'center right',
                 bbox_to_anchor = (1.05,0.5))
    
    g.savefig(os.path.join(figure_dir,'CNN_performance.jpeg'),
              dpi               = 300,
              bbox_inches       = 'tight')
    g.savefig(os.path.join(figure_dir,'CNN_performance (light).jpeg'),
    #          dpi = 300,
              bbox_inches       = 'tight')
    # g.savefig(os.path.join(paper_dir,'CNN_performance.jpeg'),
    #           dpi               = 300,
    #           bbox_inches       = 'tight')
    # g.savefig(os.path.join(paper_dir,'CNN_performance_light.jpeg'),
    # #          dpi               = 300,
    #           bbox_inches       = 'tight')
    g.savefig(os.path.join(collect_dir,'figure4.pdf'),
                dpi = 300,
              bbox_inches = 'tight')
    g.savefig(os.path.join(collect_dir,'figure4.png'),
              bbox_inches = 'tight')
    
    # plot the decoding when CNN failed
    df_stat = {'noise_level':[],
               '# of hidden units':[],
               'Dropout rate':[],
               'hidden_activation':[],
               'output_activation':[],
               'Model name':[],
               'CNN_performance':[],
               'SVM_performance':[],
               'CNN_pval':[],
               'SVM_pval':[],
               }
    for attr, df_sub in tqdm(df.groupby(['noise_level',
                                    '# of hidden units',
                                    'Dropout rate',
                                    'hidden_activation',
                                    'output_activation',
                                    'Model name',])):
        for col in ['noise_level',
                    '# of hidden units',
                    'Dropout rate',
                    'hidden_activation',
                    'output_activation',
                    'Model name']:
            df_stat[col].append(df_sub[col].values[0])
        df_stat['CNN_performance'].append(df_sub['cnn_score_mean'].values[0])
        df_stat['SVM_performance'].append(df_sub['svm_score_mean'].values[0])
        df_stat['CNN_pval'].append(df_sub['cnn_pval'].values[0])
        df_stat['SVM_pval'].append(df_sub['svm_cnn_pval'].values[0])
    
    df_stat = pd.DataFrame(df_stat)
    df_stat.to_csv(os.path.join(paper_dir,
                                'CNN_SVM_stats.csv'),index = False)

    df_chance = df_stat[df_stat['CNN_pval'] > 0.05]
    
    
    df_plot                         = df_chance.copy()#loc[idxs,:]
    df_plot['Decode Above Chance']  = df_plot['SVM_pval'] < 0.05
    df_plot = df_plot.sort_values(['# of hidden units','Dropout rate','Model name'])
    df_plot['# of hidden units'] = df_plot['# of hidden units'].astype('category')
    print(pd.unique(df_plot['# of hidden units']))
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
                    y               = 'SVM_performance',
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
                    height          = 4,
                    aspect          = 2,
                    )
    (g.set_axis_labels('Noise level','ROC AUC')
      .set_titles('')
      .set(xlim = (-0.1,50.5),ylim = (0,.8)))
    [ax.set(xticks = [0,n_noise_levels],
            xticklabels = [0,noise_levels.max()]
            ) for ax in g.axes.flatten()]
    [simpleaxes(ax) for ax in g.axes.flatten()]
    for ax_title,ax in zip(model_names,
                           g.axes.flatten()):
        ax.set(title = ax_title)
    [ax.axhline(0.5,
                linestyle           = '--',
                color               = 'black',
                alpha               = 1.,
                lw                  = 1,
                ) for ax in g.axes.flatten()]
    [ax.axvline(bins[1],
                linestyle           = '--',
                color               = 'red',
                alpha               = 1.,
                lw                  = 1,
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
    
        tiny_ax = ax.inset_axes([.7,.2,.3,.3])
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
        tiny_ax.set_xlabel('Noise level',fontsize = 18)
        tiny_ax.set_ylabel('Proportion',fontsize = 18)
        tiny_ax.set(ylim = (0,1))
        tiny_handles,tiny_labels = tiny_ax.get_legend_handles_labels()
        tiny_ax.get_legend().remove()
        simpleaxes(tiny_ax)
    
    df_proportion = pd.concat(df_proportion)
    df_proportion.to_csv(os.path.join(paper_dir.replace('figures','stats'),
                                      'CNN_chance_decode_proportion.csv'),
                         index = False)
    handles, labels                 = g.axes[0][0].get_legend_handles_labels()
    temp = []
    for item in labels:
        if item == 'drop':
            item = 'Dropout rate'
        elif item == 'hidden_units':
            item = '# of hidden units'
        temp.append(item)
    labels = temp
    [handles.append(item) for item in tiny_handles]
    [labels.append(item) for item in ['Decode Above Chance','Decode At Chance']]
    g._legend.remove()
    for ii,color in enumerate(sns.color_palette("bright")[:k]):
        handles[ii + 1]             = Patch(facecolor = color)
    g.fig.legend(handles,
                 labels,
                 loc = "center right",
                 bbox_to_anchor = (1.05,0.5))
    
    # g.fig.suptitle('Linear SVM decoding the hidden layers of CNNs that failed to descriminate living vs. nonliving',
    #                y = 1.02)
    g.savefig(os.path.join(figure_dir,'decoding_performance.jpeg'),
              dpi = 300,
              bbox_inches = 'tight')
    g.savefig(os.path.join(figure_dir,'decoding_performance (light).jpeg'),
    #          dpi = 300,
              bbox_inches = 'tight')
    # g.savefig(os.path.join(paper_dir,'decoding_performance.jpeg'),
    #           dpi = 300,
    #           bbox_inches = 'tight')
    # g.savefig(os.path.join(paper_dir,'decoding_performance_light.jpeg'),
    # #          dpi = 300,
    #           bbox_inches = 'tight')
    # g.savefig(os.path.join(collect_dir,'decoding_performance.eps'),
    #           bbox_inches = 'tight')
    g.savefig(os.path.join(collect_dir,'figure5.pdf'),
              dpi = 300,
              bbox_inches = 'tight')
    g.savefig(os.path.join(collect_dir,'figure5.png'),
              bbox_inches = 'tight')

    

















