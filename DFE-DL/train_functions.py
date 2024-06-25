import torch
import torch.nn as nn
import torch_geometric
import datetime
import numpy as np
from tqdm import tqdm
import math
from collections import OrderedDict

from itertools import chain

import random

def MSE(y, y_hat):
    MSE = torch.mean(torch.square(y - y_hat))
    return MSE

def get_local_structure_map(psi_indices):
    LS_dict = OrderedDict()
    LS_map = torch.zeros(psi_indices.shape[1], dtype = torch.long)
    v = 0
    for i, indices in enumerate(psi_indices.T):
        tupl = (int(indices[1]), int(indices[2]))
        if tupl not in LS_dict:
            LS_dict[tupl] = v
            v += 1
        LS_map[i] = LS_dict[tupl]

    alpha_indices = torch.zeros((2, len(LS_dict)), dtype = torch.long)
    for i, tupl in enumerate(LS_dict):
        alpha_indices[:,i] = torch.LongTensor(tupl)

    return LS_map, alpha_indices

def regression_loop_alpha(model, loader, optimizers, device, epoch, batch_size, training = True, auxillary_torsion_loss = 0.02):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_aux_losses = []
    batch_sizes = []
    batch_mse = []
    batch_mae = []
    
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device)

        if training:
            for opt in optimizers:
                opt.zero_grad()
        
        output1,output2 ,latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
        
        aux_loss = torch.mean(torch.abs(1.0 - phase_shift_norm.squeeze()))
        output = (output1+output2)/2
        loss = MSE(y.squeeze(), output1.squeeze()) + MSE(y.squeeze(), output2.squeeze())
        backprop_loss = loss + aux_loss*auxillary_torsion_loss
        
        mse = loss.detach()
        mae  = torch.mean(torch.abs(y.squeeze().detach() - output.squeeze().detach()))
        
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        
            for opt in optimizers:
                opt.step()
        
        batch_sizes.append(y.shape[0])
        batch_losses.append(loss.item())
        batch_aux_losses.append(aux_loss.item())
        batch_mse.append(mse.item())
        batch_mae.append(mae.item())
         
        
    return batch_losses, batch_aux_losses, batch_sizes, batch_mse, batch_mae


def evaluate_regression_loop_alpha(model, loader, device, batch_size, dataset_size):
    model.eval()
    
    all_targets = torch.zeros(dataset_size).to(device)
    all_outputs1 = torch.zeros(dataset_size).to(device)
    all_outputs2 = torch.zeros(dataset_size).to(device)
    
    start = 0
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device) 

        with torch.no_grad():
            output1,output2 ,latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
            output = (output1+output2)/2

            all_targets[start:start + y.squeeze().shape[0]] = y.squeeze()
            all_outputs1[start:start + y.squeeze().shape[0]] = output1.squeeze()
            all_outputs2[start:start + y.squeeze().shape[0]] = output2.squeeze()
            start += y.squeeze().shape[0]
       
    return all_targets.detach().cpu().numpy(), all_outputs1.detach().cpu().numpy() ,all_outputs2.detach().cpu().numpy()


