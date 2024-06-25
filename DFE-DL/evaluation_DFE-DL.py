import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import datetime
import scipy
import gzip
import math
import rdkit
import rdkit.Chem
from rdkit.Chem import TorsionFingerprints
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random
import pickle

import os
import sys
import json

from alpha_encoder import Encoder

from train_functions import regression_loop_alpha

from train_functions import evaluate_regression_loop_alpha

from train_models import train_regression_model

from datasets_samplers import MaskedGraphDataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="AP parser cannot use radical scheme, trying to use charged frag")
args = sys.argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('reading data...')
string_to_object = {
    "torch.nn.LeakyReLU(negative_slope=0.01)": torch.nn.LeakyReLU(negative_slope=0.01),
    "torch.nn.LeakyReLU()": torch.nn.LeakyReLU(),
    "torch.nn.Identity()": torch.nn.Identity(),
    "torch.nn.ReLU()": torch.nn.ReLU(),
    "torch.nn.Sigmoid()": torch.nn.Sigmoid(),
    "torch.nn.Tanh()": torch.nn.Tanh()
}
# READ HYPERPARAMETERS
with open('**/params_DFE-DL_evaluation.json') as f: # args[1] should contain path to params.json file
    params = json.load(f)

# set random seed for creating the testing set first
#full_dataframe = pd.read_pickle('**/RE dataset_one conformer.pkl')        # this can be a .pkl involving 3D conformers

#train_smiles = pd.read_csv('**/train.csv')      
#train_smiles = list(train_smiles.iloc[:,0])

#val_smiles = pd.read_csv('**/val.csv')
#val_smiles = list(val_smiles.iloc[:,0])

test_smiles = pd.read_csv('**/test.csv')
test_smiles = list(test_smiles.iloc[:,0])

#train_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in train_smiles)] 
#val_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in val_smiles)] 
test_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in test_smiles)] 

#test_dataframe = pd.read_pickle('**')  ## this can be set as an external test set

#print('the number of molecules in training set',train_dataframe.shape[0],'the number of molecules in validation set',val_dataframe.shape[0],'the number of molecules in test set',test_dataframe.shape[0])

#CREATE MODEL
seed = random.randint(1, 1000)
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

print('creating model...')
layers_dict = deepcopy(params['layers_dict'])

activation_dict = deepcopy(params['activation_dict'])
for key, value in params['activation_dict'].items(): 
    activation_dict[key] = string_to_object[value] # convert strings to actual python objects/functions using pre-defined mapping

num_node_features = 31
num_edge_features = 7

model = Encoder(
    F_z_list = params['F_z_list'], # dimension of latent space
    F_H = params['F_H'], # dimension of final node embeddings, after EConv and GAT layers
    F_H_embed = num_node_features, # dimension of initial node feature vector, currently 31
    F_E_embed = num_edge_features, # dimension of initial edge feature vector, currently 7
    F_H_EConv = params['F_H_EConv'], # dimension of node embedding after EConv layer
    layers_dict = layers_dict,
    activation_dict = activation_dict,
    GAT_N_heads = params['GAT_N_heads'],
    EConv_bias = params['EConv_bias'], 
    GAT_bias = params['GAT_bias'], 
    encoder_biases = params['encoder_biases'], 
    dropout = params['dropout'], # applied to hidden layers (not input/output layer) of Encoder MLPs, hidden layers (not input/output layer) of EConv MLP, and all GAT layers (using their dropout parameter)
    )

# LOADING PRETRAINED MODEL
if params['pretrained'] != "":              # load trained models
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(params['pretrained'], map_location=next(model.parameters()).device), strict=False)

model.to(device)

num_workers = params['num_workers']

# BUILDING DATA LOADERS
test_dataset = MaskedGraphDataset(test_dataframe, 
                                    regression = 'Reorg_energy_cation_b3lyp_def2svpp',
                                    stereoMask = False,
                                    mask_coordinates = False, 
                                    )

test_loader = torch_geometric.loader.DataLoader(test_dataset, num_workers = num_workers, batch_size = 1000, shuffle = False)

# BEGIN EVALUATION
targets, outputs1,outputs2 = evaluate_regression_loop_alpha(model, test_loader, device, batch_size = 1000, dataset_size = len(test_dataset))
outputs = (outputs1 + outputs2)/2
print('MAE: ', mean_absolute_error(targets, outputs))
print('RMSE: ', mean_squared_error(targets, outputs, squared=False))
print('R2: ', r2_score(targets, outputs))


result = pd.DataFrame(np.concatenate([targets.reshape(-1,1),outputs1.reshape(-1,1),outputs2.reshape(-1,1)],axis=1))

result.to_csv('**/outtest_predict.csv',index=False) # save the results of prediction