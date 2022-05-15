#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:29:59 2020
@author: nmei
"""
import os
import itertools
import numpy as np
import pandas as pd
from shutil import rmtree




verbose = 1
batch_size = 16
node = 1
core = 16
mem = 4 
cput = 24 
units = [2,5,10,20,50,100,300] # one unit hidden layer cannot learn
dropouts = [0,0.25,0.5,0.75]
activations = ['elu',
               'relu',
               'selu',
               'sigmoid',
               'tanh',
               'linear',
               ]
models = ['vgg19_bn','resnet50','alexnet','densenet169','mobilenet']
output_activations = ['softmax','sigmoid',]

temp = np.array(list(itertools.product(*[models,units,dropouts,activations,output_activations])))
df = pd.DataFrame(temp,columns = ['model_names','hidden_units','dropouts','hidden_activations','output_activations'])
df['hidden_units'] = df['hidden_units'].astype(int)
df['dropouts'] = df['dropouts'].astype(float)

#############
template = '9.1.hidden_layer_all_noise.py'
scripts_folder = 'all_in_all'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')

# add_on = """from shutil import copyfile
# copyfile('../utils_deep.py','utils_deep.py')
# """

from shutil import copyfile
copyfile('utils_deep.py',os.path.join(scripts_folder,'utils_deep.py'))

collections = []
first_GPU,second_GPU = [],[]
replace = False # change to second GPU

for ii,row in df.iterrows():

    src = '_{}_{}_{}_{}_{}'.format(*list(row.to_dict().values()))

    new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{src}.py').replace('1.1.train_many_models','1.1.train'))
    if ii > df.shape[0]/2 :
        replace = True
        second_GPU.append(new_scripts_name)
    else:
        first_GPU.append(new_scripts_name)
    with open(new_scripts_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "../" in line:
                    line = line.replace("../","../../")
                elif "print_train             = True" in line:
                    line = line.replace('True','False')
                elif "pretrain_model_name     = " in line:
                    line = f"pretrain_model_name     = '{row['model_names']}'\n"
                elif "hidden_units            = " in line:
                    line = f"hidden_units            = {row['hidden_units']}\n"
                elif "hidden_func_name        =" in line:
                    line = f"hidden_func_name        = '{row['hidden_activations']}'\n"
                elif "hidden_dropout          = " in line:
                    line = f"hidden_dropout          = {float(row['dropouts'])}\n"
                elif "output_activation       = " in line:
                    line = f"output_activation       = '{row['output_activations']}'\n"
                elif "train_folder            = " in line:
                    line = "train_folder            = 'greyscaled'\n"
#                elif "n_experiment_runs   = " in line:
#                    line = line.replace('20','1000')
                elif "True #" in line:
                    line = line.replace("True","False")
                # elif "import utils_deep" in line:
                #     line = "{}\n{}".format(add_on,line)
                # elif "from sklearn import metrics" in line:
                #     line = line + '\n' + add_on
                elif "idx_GPU = 0" in line:
                    if replace:
                        line = line.replace('0','-1')
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'SIM{ii+1}')
    content = f"""#!/bin/bash
#SBATCH --partition=regular
#SBATCH --job-name=SIM{ii+1}
#SBATCH --cpus-per-task={core}
#SBATCH --nodes={node}
#SBATCH --ntasks-per-node=1
#SBATCH --time={cput}:00:00
#SBATCH --mem-per-cpu={mem}G
#SBATCH --output=outputs/out_{ii+1}.txt
#SBATCH --error=outputs/err_{ii+1}.txt
#SBATCH --mail-user=nmei@bcbl.eu

source /scratch/ningmei/.bashrc
conda activate bcbl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/ningmei/anaconda3/lib
module load FSL/6.0.0-foss-2018b
cd $SLURM_SUBMIT_DIR

pwd
echo {new_scripts_name.split('/')[-1]}
python3 "{new_scripts_name.split('/')[-1]}"
    """
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"sbatch SIM{ii+1}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(0.1)\nos.system("{line}")\n')
    f.close()

# from glob import glob
# all_scripts = glob(os.path.join(scripts_folder,'simulation*.py'))

# with open(os.path.join(scripts_folder,'run_all.py'),'w') as f:
#     f.write('import os\n')
#     for files in all_scripts:
#         file_name = files.split('bash/')[-1]
#         f.write(f'os.system("python {file_name}")\n')
#     f.close()
