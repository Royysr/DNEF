import rdkit
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import numpy as np
import networkx as nx
import random
import torch

atomTypes = ['P', 'S', 'O', 'Cl', 'I', 'C', 'Si', 'Se', 'N', 'B', 'Te', 'As', 'F', 'Br','H']
formalCharge = [-1, 1, 0]
degree = [1, 2, 3, 4]

hybridization = [
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

string_to_object = {
    "torch.nn.LeakyReLU(negative_slope=0.01)": torch.nn.LeakyReLU(negative_slope=0.01),
    "torch.nn.LeakyReLU()": torch.nn.LeakyReLU(),
    "torch.nn.Identity()": torch.nn.Identity(),
    "torch.nn.ReLU()": torch.nn.ReLU(),
    "torch.nn.Sigmoid()": torch.nn.Sigmoid(),
    "torch.nn.Tanh()": torch.nn.Tanh()
}

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) 
    array_adj = np.array(np.nonzero(adj), dtype = int) 
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) 
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index

def get_all_paths(G, N = 3):
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

def getNodeFeatures(list_rdkit_atoms, owningMol):
    F_v = (len(atomTypes)+1) +\
        (len(degree)+1) + \
        (len(formalCharge)+1) +\
        (len(hybridization)+1) +\
        1
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) 
        features += one_hot_embedding(node.GetTotalDegree(), degree) 
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge) 
        features += one_hot_embedding(node.GetHybridization(), hybridization) 
        features += [int(node.GetIsAromatic())] 

        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)

def getEdgeFeatures(list_rdkit_bonds):
    F_e = (len(bondTypes)+1) + 2 # 7
    
    edge_features = np.zeros((len(list_rdkit_bonds)*2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes) 
        features += [int(edge.GetIsConjugated())] # dim=1
        features += [int(edge.IsInRing())] # dim=1  
 
        edge_features[2*edge_index: 2*edge_index+2, :] = features
        
    return np.array(edge_features, dtype = np.float32)

def getInternalCoordinatesFromAllPaths(mol, adj): 
    conformer = mol
    graph = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)
    
    distance_paths, angle_paths, dihedral_paths = get_all_paths(graph, N = 1), get_all_paths(graph, N = 2), get_all_paths(graph, N = 3)
    
    if len(dihedral_paths) == 0:
        raise Exception('No Dihedral Angle Detected')
    
    bond_distance_indices = np.array(distance_paths, dtype = int)
    bond_angle_indices = np.array(angle_paths, dtype = int)
    dihedral_angle_indices = np.array(dihedral_paths, dtype = int)


    bond_distances = np.array([rdMolTransforms.GetBondLength(conformer, int(index[0]), int(index[1])) for index in bond_distance_indices], dtype = np.float32)
    bond_angles = np.array([rdMolTransforms.GetAngleRad(conformer, int(index[0]), int(index[1]), int(index[2])) for index in bond_angle_indices], dtype = np.float32)
    dihedral_angles = np.array([rdMolTransforms.GetDihedralRad(conformer, int(index[0]), int(index[1]), int(index[2]), int(index[3])) for index in dihedral_angle_indices], dtype = np.float32)
   
    return bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices

def embedConformerWithAllPaths(rdkit_mol3D):

    mol = rdkit_mol3D
    conformer = mol.GetConformer()

    # Edge Index
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # Edge Features
    bonds = []
    for b in range(int(edge_index.shape[1]/2)):
        bond_index = edge_index[:,::2][:,b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    # Node Features 
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = getNodeFeatures(atoms, mol)
    
    bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices = getInternalCoordinatesFromAllPaths(conformer, adj)

    return atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices
