#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:19:36 2020
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

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate
from sklearn.utils import shuffle as sk_shuffle
from sklearn.inspection import permutation_importance

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
sns.set_context('poster',font_scale = 1.5,)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
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
    
    
    df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
    df                  = df.sort_values(['hidden_activation','output_activation'])
    df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']
    
    df_stat = pd.read_csv(os.path.join(paper_dir,
                                'CNN_SVM_stats.csv'))
    
    df_chance = df_stat[df_stat['CNN_pval'] > 0.05]
    
    ps = {'chance_all':0,
          'chance_high':0}
    df_chance['diff'] = df_chance['SVM_performance'].values - df_chance['CNN_performance'].values
    df_chance['labels'] = df_chance['SVM_pval'].apply(lambda x: x < 0.05)
    df_j= df_chance[df_chance['noise_level'] >= np.median(noise_levels)]
    
    a = df_chance['SVM_performance'].values
    b = df_chance['CNN_performance'].values
    diff = a - b
    t = diff.mean()
    null = diff - diff.mean()
    null_dist = np.array([np.random.choice(null,size=diff.shape[0],replace = True).mean() for _ in tqdm(range(int(1e5)))])
    p = ((np.sum(null_dist >= t)) + 1) /(null_dist.shape[0] + 1)
    ps['chance_all'] = p
    
    a = df_j['SVM_performance'].values
    b = df_j['CNN_performance'].values
    diff = a - b
    t = diff.mean()
    null = diff - diff.mean()
    null_dist = np.array([np.random.choice(null,size=diff.shape[0],replace = True).mean() for _ in tqdm(range(int(1e5)))])
    p = ((np.sum(null_dist >= t)) + 1) /(null_dist.shape[0] + 1)
    ps['chance_high'] = p
    
    fig,axes = plt.subplots(figsize = (24,12),ncols = 2)
    ax = axes[0]
    a = df_chance['SVM_performance'].values
    b = df_chance['CNN_performance'].values
    diff = a - b
    df_temp = pd.DataFrame(diff.reshape(-1,1),columns = ['SVM - CNN'])
    df_temp['x'] = 0
    df_temp['noise_level'] = df_chance['noise_level'].values
    df_temp['p < 0.05'] = df_chance['SVM_pval'].values < 0.05
    df_temp['SVM_performance'] = a
    ax = sns.violinplot(x = 'x',
                        y = 'SVM_performance',
                        # hue = 'p < 0.05',
                        data = df_temp,
                        split = True, 
                        cut = 0,
                        inner = 'quartile',
                        ax = ax,
                        )
    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.6)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('');ax.set_xticklabels([])
    ax = axes[1]
    from collections import Counter
    temp = dict(Counter(df_chance['SVM_pval'] < 0.05))
    ax.bar(0,temp[True]/np.sum(list(temp.values())),color = 'green',)
    ax.bar(1,temp[False]/np.sum(list(temp.values())),color = 'red')
    ax.set(xticks = [0,1],ylabel = 'Proportion')
    ax.set_xticklabels(['Decodable','Not decodable'])
    fig.savefig(os.path.join(paper_dir,'CNN chance noise all.jpeg'),
                dpi = 400,
                bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,'CNN chance noise all.jpeg'),
                dpi = 400,
                bbox_inches = 'tight')
    
    fig,axes = plt.subplots(figsize = (24,12),ncols = 2)
    ax = axes[0]
    a = df_j['SVM_performance'].values
    b = df_j['CNN_performance'].values
    diff = a - b
    df_temp = pd.DataFrame(diff.reshape(-1,1),columns = ['SVM - CNN'])
    df_temp['x'] = 0
    df_temp['noise_level'] = df_j['noise_level'].values
    df_temp['p < 0.05'] = df_j['SVM_pval'].values < 0.05
    df_temp['SVM_performance'] = a
    ax = sns.violinplot(x = 'x',
                        y = 'SVM_performance',
                        # hue = 'p < 0.05',
                        data = df_temp,
                        split = True, 
                        cut = 0,
                        inner = 'quartile',
                        ax = ax,
                        )
    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.6)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('');ax.set_xticklabels([])
    ax = axes[1]
    from collections import Counter
    temp = dict(Counter(df_j['SVM_pval'] < 0.05))
    ax.bar(0,temp[True]/np.sum(list(temp.values())),color = 'green',)
    ax.bar(1,temp[False]/np.sum(list(temp.values())),color = 'red')
    ax.set(xticks = [0,1],ylabel = 'Proportion')
    ax.set_xticklabels(['Decodable','Not decodable'])
    fig.savefig(os.path.join(paper_dir,'CNN chance noise high.jpeg'),
                dpi = 400,
                bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,'CNN chance noise high.jpeg'),
                dpi = 400,
                bbox_inches = 'tight')
    # sns.relplot(x = 'noise_level',y = 'diff',hue = 'labels',data = df_chance)
    
    labels = []
    for ii, row in tqdm(df_chance.iterrows(),desc = 'generate labels'):
        cnn_pval = row['CNN_pval']
        svm_pval = row['SVM_pval']
        if (cnn_pval > 0.05) and (svm_pval < 0.05): # target condition
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    
    col_features = ['# of hidden units','hidden_activation','output_activation',
                    'noise_level','Dropout rate','Model name']
    
    hidden_activation = preprocessing.LabelEncoder().fit(df_chance['hidden_activation'].values)
    df_chance['hidden_activation'] = hidden_activation.transform(df_chance['hidden_activation'].values)
    output_activation = preprocessing.LabelEncoder().fit(df_chance['output_activation'].values)
    df_chance['output_activation'] = output_activation.transform(df_chance['output_activation'].values)
    model_name = preprocessing.LabelEncoder().fit(df_chance['Model name'].values)
    df_chance['Model name'] = model_name.transform(df_chance['Model name'].values)
    features = df_chance[col_features].values
    
    X,y = sk_shuffle(features,labels)
    
    cv = StratifiedShuffleSplit(n_splits = 100,test_size = 0.2, random_state = 12345)
    rf = RandomForestClassifier(random_state = 12345,class_weight = 'balanced',n_jobs = 1)
    # logit = LogisticRegression(random_state = 12345,class_weight = 'balanced')
    pipeline = make_pipeline(preprocessing.StandardScaler(),
                             rf)
    cross_res = cross_validate(pipeline,
                               X,
                               y,
                               cv = cv,
                               return_estimator = True,
                               n_jobs = -1,
                               verbose = 1,
                               )
    
    importance = {name:[] for name in col_features}
    for _clf,(idx_train,idx_test) in tqdm(zip(cross_res['estimator'],cv.split(X,y)),desc='feature importance'):
        X_,y_ = X[idx_test],y[idx_test]
        clf = _clf.steps[-1][-1]
        scaler = preprocessing.StandardScaler()
        scaler.fit(X[idx_train])
        X_ = scaler.transform(X_)
        permutation_res = permutation_importance(clf,X_,y_,n_jobs = -1,random_state = 12345)
        [importance[col_name].append(value) for col_name,value in zip(col_features,permutation_res['importances_mean'])]
    importance = pd.DataFrame(importance)
    df_plot = pd.melt(importance,id_vars=None,value_vars=col_features,)
    
    df_plot.columns = ['Attributes','Feature Importance']
    
    fig,ax = plt.subplots(figsize = (16,16))
    ax = sns.violinplot(y = 'Attributes',x = 'Feature Importance',data= df_plot,ax = ax,
                        inner = 'quartile',cut = 0,
                        order = ['noise_level',
                                 'Model name',
                                 '# of hidden units',
                                 'hidden_activation',
                                 'Dropout rate',
                                 'output_activation',])
    ax.set_yticklabels(['Noise levels',
                          'CNN backbone',
                          '# of hidden units',
                          'Hidden activation function',
                          'Dropout rate',
                          'Output activation function',],
                       rotation = 0,)
    df_plot.to_csv(os.path.join(paper_dir.replace('figures','stats'),
                                'feature_importance.csv'),index=False)
    fig.savefig(os.path.join(figure_dir,'feature importance.jpeg'),dpi = 300,bbox_inches = 'tight')
    fig.savefig(os.path.join(paper_dir,'feature importance.jpeg'),dpi = 300,bbox_inches = 'tight')

# from collections import Counter
# from scipy import stats
# x = np.linspace(0, 1, 100)
# alpha,beta = 1.001,1.001
# aspects = {'hidden_units':.025,
#         'hidden_activation':.03,
#         'output_activation':.01,
#         'noise_level':0.2,
#         'drop':.02,
#         'model_name':.02}
# fig,axes = plt.subplots(figsize = (25*2,8*3),
#                       nrows = 2,
#                       ncols = 3)
# for ax,(groupby,factor_name) in zip(axes.flatten(),{'hidden_units':'Hidden Units',
#                             'hidden_activation':'Hidden Activation Function',
#                             'output_activation':'Output Activation Function',
#                             'noise_level':'Noise Level',
#                             'drop':'Dropout Rate',
#                             'model_name':'Model Architecture'}.items()):
#     x_plot = []
#     y_plot = []
#     aspect = aspects[groupby]
#     for noise_level,df_sub in df_chance.groupby([groupby]):
#         if df_sub.shape[0] > 5:
#             labels = df_sub['labels'].values.astype(int)
#             counts = dict(Counter(labels))
#             y = stats.beta.pdf(x,counts[1] + alpha, counts[0] + beta)
#             y /= np.linalg.norm(y)
#             x_plot.append(noise_level)
#             y_plot.append(y)
#     x_plot = np.array(x_plot)
#     y_plot = np.array(y_plot)


#     im = ax.imshow(y_plot.T,
#                    origin = 'lower',
#                    aspect = aspect,
#                    cmap = plt.cm.coolwarm,
#                    interpolation = 'hamming',
#                    vmin = -.15,
#                    vmax = .15,)
#     if groupby == 'noise_level':
#         xticks = np.concatenate([np.arange(x_plot.shape[0])[::8],[np.arange(x_plot.shape[0])[-1]]])
#         xticklabels = [f'{item:.2f}' for item in np.concatenate([x_plot[::8],[int(x_plot[-1])]])]
#     elif groupby == 'hidden_units':
#         xticks = np.arange(pd.unique(df_chance[groupby]).shape[0])
#         xticklabels = pd.unique(df_chance[groupby])
#     elif groupby == 'drop':
#         xticks = np.arange(pd.unique(df_chance[groupby]).shape[0])
#         xticklabels = pd.unique(df_chance[groupby])
#     elif groupby == 'hidden_activation':
#         xticks = np.arange(pd.unique(df_chance[groupby]).shape[0])
#         xticklabels = pd.unique(df_chance[groupby])
#     elif groupby == 'output_activation':
#         xticks = np.arange(pd.unique(df_chance[groupby]).shape[0])
#         xticklabels = pd.unique(df_chance[groupby])
#     elif groupby == 'model_name':
#         xticks = np.arange(pd.unique(df_chance[groupby]).shape[0])
#         xticklabels = pd.unique(df_chance[groupby])

#     output_activation
#     ax.set(xticks = xticks,
#            # xticklabels = xticklabels,
#            yticks = [0,25,50,75,100],
#            yticklabels = [0,0.25,0.5,0.75,1],
#            xlabel = factor_name,
#            ylabel = 'Probability',
#            )
#     ax.set_xticklabels(xticklabels,rotation = 45,ha = 'center')
# fig.tight_layout()
# fig.colorbar(im, ax=axes.ravel().tolist())
# fig.suptitle('Probablity of the SVM being able to decode the CNN when CNN performs at chance level')
# fig.savefig(os.path.join(figure_dir,'Probablity of the SVM being able to decode the CNN when CNN performs at chance level,jpeg'),
#             dpi = 400,
#             bbox_inches = 'tight')
# fig.savefig(os.path.join(paper_dir,'Probablity.png'),
#             dpi = 400,
#             bbox_inches = 'tight')