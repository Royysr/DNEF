import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import datetime
import scipy
import gzip
import math
import rdkit

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import rdkit.Chem
from rdkit.Chem import TorsionFingerprints
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="AP parser cannot use radical scheme, trying to use charged frag")

import os
import sys
import json

from model.alpha_encoder import Encoder

from model.train_functions import regression_loop_alpha

from model.train_functions import evaluate_regression_loop_alpha

from model.train_models import train_regression_model

from model.datasets_samplers import MaskedGraphDataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

import sklearn

args = sys.argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
string_to_object = {
    "torch.nn.LeakyReLU(negative_slope=0.01)": torch.nn.LeakyReLU(negative_slope=0.01),
    "torch.nn.LeakyReLU()": torch.nn.LeakyReLU(),
    "torch.nn.Identity()": torch.nn.Identity(),
    "torch.nn.ReLU()": torch.nn.ReLU(),
    "torch.nn.Sigmoid()": torch.nn.Sigmoid(),
    "torch.nn.Tanh()": torch.nn.Tanh()
}
print('reading data...')

with open('**/params_DFE-DL_training') as f: # args[1] should contain path to params.json file
    params = json.load(f)

test_seed = random.randint(1, 1000)       # set random seed for creating the testing set first
random.seed(test_seed)
full_dataframe = pd.read_pickle('**/new_RE.pkl')

train_smiles = pd.read_csv('**/train.csv')      
train_smiles = list(train_smiles.iloc[:,0])

val_smiles = pd.read_csv('**/val.csv')
val_smiles = list(val_smiles.iloc[:,0])

test_smiles = pd.read_csv('**/test.csv')
test_smiles = list(test_smiles.iloc[:,0])

train_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in train_smiles)] 
val_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in val_smiles)] 
test_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in test_smiles)] 

print('the number of molecules in training set',train_dataframe.shape[0],'the number of molecules in validation set',val_dataframe.shape[0],'the number of molecules in test set',test_dataframe.shape[0])

# CREATE DIRECTORY FOR SAVING/CHECKPOINTING
save = params['save']

PATH = '**/**' # should contain path to subfolder where files will be saved
if PATH[-1] != '/':
    PATH = PATH + '/'

if not os.path.exists(PATH) and save == True:
    os.makedirs(PATH)

# CREATE MODEL
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
    chiral_message_passing = params['chiral_message_passing'],
    c_coefficient_normalization = params['c_coefficient_normalization'], # None, or one of ['softmax']
    encoder_reduction = params['encoder_reduction'], #mean or sum
    output_concatenation_mode = params['output_concatenation_mode'], 
    EConv_bias = params['EConv_bias'], 
    GAT_bias = params['GAT_bias'], 
    encoder_biases = params['encoder_biases'], 
    dropout = params['dropout'], # applied to hidden layers (not input/output layer) of Encoder MLPs, hidden layers (not input/output layer) of EConv MLP, and all GAT layers (using their dropout parameter)
    )

model.to(device)

# SET UNLEARNABLE PARAMETERS

print('learning phase shifts...')

# DEFINE OPTIMIZERS AND SCHEDULER
lr = params['default_lr']
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8, min_lr=1e-6)
optimizers = [optimizer]

# BUILDING DATA LOADERS
batch_size = params['batch_size']

train_dataset = MaskedGraphDataset(train_dataframe, 
                                    regression = 'Reorg_energy_cation_b3lyp_def2svpp',
                                    stereoMask = False,
                                    mask_coordinates = False, 
                                    )

val_dataset = MaskedGraphDataset(val_dataframe, 
                                    regression = 'Reorg_energy_cation_b3lyp_def2svpp',
                                    stereoMask = False,
                                    mask_coordinates = False, 
                                    )

num_workers = params['num_workers']
train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True)
val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True)


# BEGIN TRAINING
if not os.path.exists(PATH + 'checkpoint_models') and save == True:
    os.makedirs(PATH + 'checkpoint_models')

N_epochs = params['N_epochs']
auxillary_torsion_loss = params['auxillary_torsion_loss']

best_state_dict = train_regression_model(model, 
                           train_loader, 
                           val_loader,
                           N_epochs = N_epochs, 
                           optimizers = optimizers, 
                           scheduler = scheduler,
                           device = device, 
                           batch_size = batch_size, 
                           auxillary_torsion_loss = auxillary_torsion_loss,
                           save = save,
                           PATH = PATH)

print('completed training')

