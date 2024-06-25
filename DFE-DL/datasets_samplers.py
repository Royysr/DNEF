import torch
import torch.nn as nn
import torch_geometric

import math
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import chain

import rdkit
from rdkit.Chem import MACCSkeys,AllChem,rdMolDescriptors,rdmolops,Descriptors,TorsionFingerprints,rdMolTransforms
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
import networkx as nx

from tqdm import tqdm
import datetime
import random

from ocelot.schema.graph import MolGraph,BasicGraph

from embedding_functions import embedConformerWithAllPaths
def get_fps(mol):
    fps = []
    estate = []
    maccs = MACCSkeys.GenMACCSKeys(mol)
    fps0 = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    fps2 = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    fps5 = Fingerprinter.FingerprintMol(mol)[0]
    fps6 = Chem.RDKFingerprint(mol,minPath=2,maxPath=2,fpSize=2048)
    fps.append(maccs+fps0+fps2+fps6)
    estate.append(fps5)
    fps = np.array(fps)
    fps = np.concatenate([fps,np.vstack(estate)],axis=1)
    return fps

def get_chrom_des(mol):
    mg = MolGraph.from_rdmol(mol)
    try:
        backbone, fragments = mg.partition('chrom')     
        b = BasicGraph(backbone)
        try:
            bm = b.to_rdmol(charged_fragments=True)[0]                     
            chro_smi = Chem.MolToSmiles(bm)
            mol = Chem.AddHs(Chem.MolFromSmiles(chro_smi))                 
            charge = rdmolops.GetFormalCharge(mol)                       
            rotate_num = rdMolDescriptors.CalcNumRotatableBonds(mol)  
            atom_num = mol.GetNumAtoms()                       
            mol_weight = Descriptors.ExactMolWt(mol)           
            ring_num = mol.GetRingInfo().NumRings()            
        except:
            charge,rotate_num,atom_num,mol_weight,ring_num = 0,0,0,0,0
    except:
        charge,rotate_num,atom_num,mol_weight,ring_num  = 0,0,0,0,0
    return torch.as_tensor(charge,dtype=torch.float) , torch.as_tensor(rotate_num,dtype=torch.float) ,torch.as_tensor(atom_num,dtype=torch.float) ,torch.as_tensor(mol_weight,dtype=torch.float),torch.as_tensor(ring_num,dtype=torch.float)

class MaskedGraphDataset(torch_geometric.data.Dataset):
    def __init__(self, df, regression = '', stereoMask = True, mask_coordinates = False):
        super(MaskedGraphDataset, self).__init__()
        self.df = df
        self.stereoMask = stereoMask
        self.mask_coordinates = mask_coordinates
        self.regression = regression
        
    def get_all_paths(self, G, N = 3):
        # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        def findPaths(G,u,n):
            if n==0:
                return [[u]]
            paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
            return paths
    
        allpaths = []
        for node in G:
            allpaths.extend(findPaths(G,node,N))
        return allpaths
    
    def process_mol(self, mol):
        
        atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(mol)
        
        bond_angles = bond_angles % (2*np.pi)
        dihedral_angles = dihedral_angles % (2*np.pi)
        
        data = torch_geometric.data.Data(x = torch.as_tensor(node_features), edge_index = torch.as_tensor(edge_index, dtype=torch.long), edge_attr = torch.as_tensor(edge_features))
        data.bond_distances = torch.as_tensor(bond_distances)
        data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
        data.bond_angles = torch.as_tensor(bond_angles)
        data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
        data.dihedral_angles = torch.as_tensor(dihedral_angles)
        data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T
        data.fps = torch.as_tensor(get_fps(mol), dtype=torch.float)     
        data.charge,data.rotate_num,data.atom_num,data.mol_weight,data.ring_num = get_chrom_des(mol)
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol)
        
        data = self.process_mol(mol)
        
        if self.regression != '':
            y = torch.tensor(deepcopy(self.df.iloc[key][self.regression])) 

        if self.stereoMask:
            data.x[:, -9:] = 0.0
            data.edge_attr[:, -7:] = 0.0

        if self.mask_coordinates:
            data.bond_distances[:] = 0.0
            data.bond_angles[:] = 0.0
            data.dihedral_angles[:] = 0.0

        return (data, y) if self.regression != '' else data

