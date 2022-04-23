import math
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, Parameter

from utilities import direct_sum

class MultiLayerPerceptron(Module):

    def __init__(self, layer_dims: List[int], layer_activations: List[Module]) -> None:
        super().__init__()
        if not len(layer_dims) - 1 == len(layer_activations):
            raise ValueError("len(layer_dims) - 1 must equal len(layer_activations)")
        
        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            modules.append(layer_activations[i])

        self.layers = nn.Sequential(*modules)
        self.layers.apply(MultiLayerPerceptron.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


class NeuroSATAssign(Module):

    def __init__(self, embedding_dim: int, iterations: int, mlp_n_layers: int, 
                    mlp_activation: Module=nn.ReLU(), lstm_activation: Module=nn.Tanh()) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.iterations = iterations

        self.L_init = torch.normal(0, 1, (1,embedding_dim), requires_grad=True) / math.sqrt(embedding_dim)
        self.C_init = torch.normal(0, 1, (1,embedding_dim), requires_grad=True) / math.sqrt(embedding_dim)

        layer_dims = [embedding_dim] * (mlp_n_layers + 1)
        layer_activations = ([mlp_activation] * (mlp_n_layers - 1)) + [nn.Identity()] 
        self.LC_msg = MultiLayerPerceptron(layer_dims, layer_activations)
        self.CL_msg = MultiLayerPerceptron(layer_dims, layer_activations)

        self.L_update = LayerNormLSTMCell(input_size=embedding_dim * 2, hidden_size=embedding_dim, activation=lstm_activation)
        self.C_update = LayerNormLSTMCell(input_size=embedding_dim, hidden_size=embedding_dim, activation=lstm_activation)

        self.L_readout = MultiLayerPerceptron([embedding_dim * 2, embedding_dim, embedding_dim, 1], [nn.ReLU(), nn.ReLU(), nn.Sigmoid()])

    def forward(self, adjacency_matrix: Tensor, batch_lit_counts: Tensor, device=torch.device("cpu")):
        L_embeddings = torch.tile(self.L_init, (adjacency_matrix.shape[0],1)).to(device) # N x D
        C_embeddings = torch.tile(self.L_init, (adjacency_matrix.shape[1],1)).to(device) # M x D

        L_state_init = (L_embeddings, torch.zeros((adjacency_matrix.shape[0], self.embedding_dim)).to(device)) 
        C_state_init = (C_embeddings, torch.zeros((adjacency_matrix.shape[1], self.embedding_dim)).to(device))

        flip = NeuroSATAssign.batched_flip(batch_lit_counts).to(device)
        
        L_state_final, C_state_final = self.iterate(adjacency_matrix, L_state_init, C_state_init, flip)

        lit_readouts = torch.cat((L_state_final[0], torch.matmul(flip, L_state_final[0])), dim=1)
        lit_readouts = NeuroSATAssign.pos_lit_select(lit_readouts, batch_lit_counts)
        lit_readouts = self.L_readout(lit_readouts) # B x N x 2D
        return lit_readouts, [batch_n // 2 for batch_n in batch_lit_counts]

    def iterate(self, adjacency_matrix, L_state_init, C_state_init, flip_mtrx):
        L_state = L_state_init
        C_state = C_state_init
        
        for itr in range(self.iterations):
            LC_messages = torch.matmul(adjacency_matrix.transpose(0,1), self.LC_msg(L_state[0]))
            _, C_state = self.C_update(input=LC_messages, state=C_state)

            CL_messages = torch.matmul(adjacency_matrix, self.CL_msg(C_state[0])) 
            _, L_state = self.L_update(input=torch.cat((CL_messages, torch.matmul(flip_mtrx, L_state[0])), dim=1), state=L_state)
        return L_state, C_state

    @staticmethod
    def batched_flip(batch_counts) -> Tensor: # TODO: Modify to handle batches
        flip_matrices = []
        for batch_lits in batch_counts:
            n = batch_lits
            eye = torch.eye(n)
            flip_eye = torch.cat((eye[n//2:n], eye[0:n//2]))
            flip_matrices.append(flip_eye)
        permute = direct_sum(flip_matrices)
        return permute

    @staticmethod
    def pos_lit_select(lit_embeddings, batch_counts) -> Tensor:
        mask = []
        for b_n in batch_counts:
            mask.extend([True] * (b_n // 2))
            mask.extend([False] * (b_n // 2))
        mask = torch.tensor(mask)
        return lit_embeddings[mask]
            

class LayerNormLSTMCell(Module):

    def __init__(self, input_size:int, hidden_size: int, activation: Module=nn.Tanh()) -> None:
        super().__init__()
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.activation = activation

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def forward(self, input:Tensor, state:Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.matmul(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.matmul(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * self.activation(cy)

        return hy, (hy, cy)


class NeuroSATLoss(Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, predictions, lit_sizes, discriminators):
        losses = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        lit_index = 0
        for lits_size, discrim in zip(lit_sizes, discriminators):
            pred = predictions[0 + lit_index:lits_size + lit_index]
            lit_index += lits_size
            signal = discrim(pred)
            signal = (signal * 2 - 1) * -1
            loss = torch.multiply(signal, torch.sum(torch.square(torch.sub(pred, 0.5)))) # sat * (x - 0.5)^2
            loss = torch.divide(loss, lits_size) # normalize by problem size
            losses = torch.add(losses, loss)
        losses = torch.divide(losses, len(lit_sizes)) # normalize by batch size
        return losses


if __name__ == "__main__":
    pass
