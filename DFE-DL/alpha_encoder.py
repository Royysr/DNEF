import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
import torch_scatter
import math

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation_hidden, activation_out, biases, dropout):
        super(MLP, self).__init__()
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.dropout = dropout

        if len(hidden_sizes) > 0:
            self.linear_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0], bias = biases)])
            self.linear_layers.extend([nn.Linear(in_size, out_size, bias = biases) 
                                       for (in_size, out_size) 
                                       in zip(hidden_sizes[0:-1], (hidden_sizes[1:]))])
            self.linear_layers.append(nn.Linear(hidden_sizes[-1], output_size, bias = biases))
        
        else:
            self.linear_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias = biases)])
        
    def forward(self, x):
        if len(self.linear_layers) == 1:
            out = self.activation_out(self.linear_layers[0](x))
        
        else:
            out = self.activation_hidden(self.linear_layers[0](x))
            for i, layer in enumerate(self.linear_layers[1:-1]):
                out = self.activation_hidden(layer(out))
                out = F.dropout(out, p = self.dropout, training = self.training)
            out = self.activation_out(self.linear_layers[-1](out))

        return out

class GraphNodeEmbedder(nn.Module):
    def __init__(self, EConv_in_channels, EConv_out_channels, EConv_mlp, GAT_out_channels, GAT_N_heads, EConv_bias, GAT_bias, dropout):
        super(GraphNodeEmbedder, self).__init__()
        
        self.EConv = torch_geometric.nn.NNConv(
            in_channels = EConv_in_channels,
            out_channels = EConv_out_channels,
            nn = EConv_mlp,
            aggr = "add",
            root_weight = True,
            bias = EConv_bias
        )                                                                           
        
        self.GAT_layers = nn.ModuleList([
            torch_geometric.nn.GATConv(
                in_channels = EConv_out_channels,
                out_channels = GAT_out_channels[0],
                heads = GAT_N_heads,
                concat=False,
                negative_slope = 0.2,
                dropout = dropout,
                add_self_loops = True,
                bias = GAT_bias,
                flow="source_to_target"
            )
        ])
        
        for i in range(1, len(GAT_out_channels)):
            self.GAT_layers.append(
                torch_geometric.nn.GATConv(
                    in_channels = GAT_out_channels[i-1],
                    out_channels = GAT_out_channels[i],
                    heads = GAT_N_heads,
                    concat=False,
                    negative_slope = 0.2,
                    dropout = dropout,
                    add_self_loops = True,
                    bias = GAT_bias,
                    flow="source_to_target"
                )
            )

    def forward(self, x, edge_index, edge_attr):
        out = self.EConv(x, edge_index, edge_attr)
        for i, layer in enumerate(self.GAT_layers):
            out = layer(out, edge_index)
        return out

class InternalCoordinateEncoder(nn.Module):
    def __init__(self, F_z_list, F_H, hidden_sizes_D, hidden_sizes_phi, hidden_sizes_c, hidden_sizes_sinusoidal_shift, hidden_sizes_alpha, activation_dict, biases, dropout):
        super(InternalCoordinateEncoder, self).__init__()
        
        self.F_z_list = F_z_list
        self.F_H = F_H

        self.Encoder_D = MLP(
            input_size = 2*F_H + 1,
            output_size = self.F_z_list[0],
            hidden_sizes = hidden_sizes_D,
            activation_hidden = activation_dict['encoder_hidden_activation_D'],
            activation_out = activation_dict['encoder_output_activation_D'],
            biases = biases,
            dropout = dropout,
            )

        self.Encoder_phi = MLP(
            input_size = 3*F_H + 2, 
            output_size = self.F_z_list[1],
            hidden_sizes = hidden_sizes_phi,
            activation_hidden = activation_dict['encoder_hidden_activation_phi'],
            activation_out = activation_dict['encoder_output_activation_phi'],
            biases = biases,
            dropout = dropout,
            )
        
        self.Encoder_c = MLP(
            input_size = 4*F_H,
            output_size = 1,
            hidden_sizes = hidden_sizes_c,
            activation_hidden = activation_dict['encoder_hidden_activation_c'],
            activation_out = activation_dict['encoder_output_activation_c'],
            biases = biases,
            dropout = dropout,
            )
        
        self.Encoder_sinusoidal_shift = MLP(
            input_size = 4*F_H,
            output_size = 2,
            hidden_sizes = hidden_sizes_sinusoidal_shift,
            activation_hidden = activation_dict['encoder_hidden_activation_sinusoidal_shift'],
            activation_out = activation_dict['encoder_output_activation_sinusoidal_shift'],
            biases = biases,
            dropout = dropout,
            )
        
        alpha_input_size = 2*F_H + 1
        self.Encoder_alpha = MLP(
            input_size = alpha_input_size,
            output_size = self.F_z_list[2],
            hidden_sizes = hidden_sizes_alpha,
            activation_hidden = activation_dict['encoder_hidden_activation_alpha'],
            activation_out = activation_dict['encoder_output_activation_alpha'],
            biases = biases,
            dropout = dropout,
            )

    def forward(self, H_embeddings, distances, distance_indices, phis, phi_indices, psis, psi_indices, node_map, LS_map, alpha_indices):

        i,j = distance_indices
        hi_hj_d_tensor_forward = torch.cat([H_embeddings[i], H_embeddings[j], distances.unsqueeze(1)], dim=1)
        hi_hj_d_tensor_reverse = torch.cat([H_embeddings[j], H_embeddings[i], distances.unsqueeze(1)], dim=1)
        z_D = self.Encoder_D(hi_hj_d_tensor_forward) + self.Encoder_D(hi_hj_d_tensor_reverse)

        i,j,k = phi_indices
        hi_hj_hk_phi_tensor_forward = torch.cat([H_embeddings[i], H_embeddings[j], H_embeddings[k], torch.cos(phis.unsqueeze(1)), torch.sin(phis.unsqueeze(1))], dim=1)
        hi_hj_hk_phi_tensor_reverse = torch.cat([H_embeddings[k], H_embeddings[j], H_embeddings[i], torch.cos(phis.unsqueeze(1)), torch.sin(phis.unsqueeze(1))], dim=1)
        z_phi = self.Encoder_phi(hi_hj_hk_phi_tensor_forward) + self.Encoder_phi(hi_hj_hk_phi_tensor_reverse)
        
        i,j,k,l = psi_indices
        hi_hj_hk_hl_tensor_forward = torch.cat([H_embeddings[i], H_embeddings[j], H_embeddings[k], H_embeddings[l]], dim = 1)
        hi_hj_hk_hl_tensor_reverse = torch.cat([H_embeddings[l], H_embeddings[k], H_embeddings[j], H_embeddings[i]], dim = 1)
        
        c_tensor = self.Encoder_c(hi_hj_hk_hl_tensor_forward) + self.Encoder_c(hi_hj_hk_hl_tensor_reverse)
        
        phase_shift_linear_output = (self.Encoder_sinusoidal_shift(hi_hj_hk_hl_tensor_forward) + self.Encoder_sinusoidal_shift(hi_hj_hk_hl_tensor_reverse)) # [linear_cos, linear_sin]

        phase_shift_norm = torch.linalg.norm(phase_shift_linear_output, ord=2, dim=1, keepdim=True)
        phase_shift_linear_output_normalized = phase_shift_linear_output / torch.max(phase_shift_norm, torch.cuda.FloatTensor(1).fill_(1e-12).squeeze()) if torch.cuda.is_available() else phase_shift_linear_output / torch.max(phase_shift_norm, torch.FloatTensor(1).fill_(1e-12).squeeze())

        phase_cos = phase_shift_linear_output_normalized[:,0]
        phase_sin = phase_shift_linear_output_normalized[:,1]
        
        normalized_c_tensor = torch.sigmoid(c_tensor.squeeze()).unsqueeze(1)
            
        scaled_torsions = torch.cat([torch.cos(psis.unsqueeze(1))*phase_cos.unsqueeze(1) - torch.sin(psis.unsqueeze(1))*phase_sin.unsqueeze(1), torch.sin(psis.unsqueeze(1))*phase_cos.unsqueeze(1) + torch.cos(psis.unsqueeze(1))*phase_sin.unsqueeze(1)], dim = 1) * normalized_c_tensor
        

        
        pooled_sums = torch_geometric.nn.global_add_pool(scaled_torsions, LS_map)
        
        x,y = alpha_indices
        radii = torch.linalg.norm(pooled_sums, ord=2, dim=1, keepdim=False)
        hx_hy_alpha_tensor_forward = torch.cat([H_embeddings[x], H_embeddings[y], radii.unsqueeze(1)], dim=1)
        hx_hy_alpha_tensor_reverse = torch.cat([H_embeddings[y], H_embeddings[x], radii.unsqueeze(1)], dim=1) 
        
        z_alpha = self.Encoder_alpha(hx_hy_alpha_tensor_forward) + self.Encoder_alpha(hx_hy_alpha_tensor_reverse)
        z_alpha_shape = z_alpha.shape
        
        z_D_pooled = torch_geometric.nn.global_add_pool(z_D, node_map[distance_indices[0]])
        z_phi_pooled = torch_geometric.nn.global_add_pool(z_phi, node_map[phi_indices[0]])
        z_alpha_pooled = torch_geometric.nn.global_add_pool(z_alpha, node_map[alpha_indices[0]])
        
        z = torch.cat([z_D_pooled, z_phi_pooled, z_alpha_pooled], dim = 1)
        
        return z, phase_shift_norm, z_alpha, c_tensor, phase_cos, phase_sin, torch.cat([torch.cos(psis.unsqueeze(1)), torch.sin(psis.unsqueeze(1))], dim = 1), pooled_sums


class Encoder(nn.Module):
    def __init__(self, F_z_list, F_H, F_H_embed, F_E_embed, F_H_EConv, layers_dict, activation_dict, GAT_N_heads = 1, chiral_message_passing = False, CMP_EConv_MLP_hidden_sizes = [64], CMP_GAT_N_layers = 2, CMP_GAT_N_heads = 1, c_coefficient_normalization = None, encoder_reduction = 'mean', output_concatenation_mode = 'none', EConv_bias = True, GAT_bias = True, encoder_biases = True, dropout = 0.0):
        super(Encoder, self).__init__()

        assert len(F_z_list) == 3
        
        self.chiral_message_passing = chiral_message_passing
        self.output_concatenation_mode = output_concatenation_mode
        self.F_z_list = F_z_list
        self.F_H = F_H

        EConv_mlp = MLP(
            input_size = F_E_embed,
            output_size = F_H_embed*F_H_EConv,
            hidden_sizes = layers_dict['EConv_mlp_hidden_sizes'],
            activation_hidden = activation_dict['EConv_mlp_hidden_activation'],
            activation_out = activation_dict['EConv_mlp_output_activation'],
            biases = EConv_bias,
            dropout = dropout)

        GAT_out_channels = layers_dict['GAT_hidden_node_sizes']
        GAT_out_channels.append(self.F_H)

        self.Graph_Embedder = GraphNodeEmbedder(
            EConv_in_channels = F_H_embed,
            EConv_out_channels = F_H_EConv,
            EConv_mlp = EConv_mlp,
            GAT_out_channels = GAT_out_channels,
            GAT_N_heads = GAT_N_heads,
            EConv_bias = EConv_bias,
            GAT_bias = GAT_bias,
            dropout = dropout,
            )

        self.InternalCoordinateEncoder = InternalCoordinateEncoder(
            F_z_list = self.F_z_list,
            F_H = self.F_H,
            hidden_sizes_D = layers_dict['encoder_hidden_sizes_D'],
            hidden_sizes_phi = layers_dict['encoder_hidden_sizes_phi'],
            hidden_sizes_c = layers_dict['encoder_hidden_sizes_c'],
            hidden_sizes_alpha = layers_dict['encoder_hidden_sizes_alpha'],
            hidden_sizes_sinusoidal_shift = layers_dict['encoder_hidden_sizes_sinusoidal_shift'],
            activation_dict = activation_dict,  
            biases = encoder_biases,
            dropout = dropout,
            )
        

        mlp_input_size = self.F_H + sum(self.F_z_list)
        
        self.Output_MLP = MLP(
            input_size = mlp_input_size + 5,  # the dimension of skeletons descriptors
            output_size = 1, 
            hidden_sizes = layers_dict['output_mlp_hidden_sizes'],
            activation_hidden = activation_dict['output_mlp_hidden_activation'],
            activation_out = activation_dict['output_mlp_output_activation'],
            biases = encoder_biases,
            dropout = dropout)

        # FCDNN
        self.fps_MLP = MLP(         
                input_size = 6390,
                output_size = 1, 
                hidden_sizes = layers_dict['output_mlp_hidden_sizes'],
                activation_hidden = activation_dict['output_mlp_hidden_activation'],
                activation_out = activation_dict['output_mlp_output_activation'],
                biases = encoder_biases,
                dropout = dropout)
    
    def forward(self, data, LS_map, alpha_indices):

        node_features, edge_index, edge_attr, distances, distance_indices, phis, phi_indices, psis, psi_indices = data.x, data.edge_index, data.edge_attr, data.bond_distances, data.bond_distance_index, data.bond_angles, data.bond_angle_index, data.dihedral_angles, data.dihedral_angle_index
        node_map = data.batch
        
        H_embeddings = self.Graph_Embedder(node_features, edge_index, edge_attr)
       
        latent_vector, phase_shift_norm, z_alpha, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = self.InternalCoordinateEncoder(H_embeddings, distances, distance_indices, phis, phi_indices, psis, psi_indices, node_map, LS_map, alpha_indices)
        
        mol_embedding = torch_geometric.nn.global_add_pool(H_embeddings, node_map)

        embedding = torch.cat((mol_embedding, latent_vector,data.charge.unsqueeze(1),data.rotate_num.unsqueeze(1),data.atom_num.unsqueeze(1),data.mol_weight.unsqueeze(1),data.ring_num.unsqueeze(1)), dim = 1)
        
        output1 = self.Output_MLP(embedding)
        output2 = self.fps_MLP(data.fps)

        return output1,output2 ,latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha
