import math
import yaml
import json
from types import SimpleNamespace
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utilities import direct_sum, relax_sympy
from relaxations import relaxation_types
from dataset import NeuroSATDataset, collate_adjacencies

activations = {
    'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(),
    'identity': nn.Identity()
}

class MultiLayerPerceptron(Module):

    def __init__(self, layer_dims: List[int], layer_activations: List[Module], p_dropout=0.0) -> None:
        super().__init__()
        if not len(layer_dims) - 1 == len(layer_activations):
            raise ValueError("len(layer_dims) - 1 must equal len(layer_activations)")
        
        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:
                modules.append(nn.Dropout(p=p_dropout))
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

    def __init__(self, config) -> None:
        super().__init__()
        embedding_dim = config.model.embedding_dim
        mlp_n_layers = config.model.mlp_hidden_layers
        mlp_activation = activations[config.model.mlp_activation]
        lstm_activation = activations[config.model.lstm_activation]

        self.embedding_dim = embedding_dim
        self.iterations = config.model.lstm_iters

        self.L_init = Parameter(torch.normal(0, 1, (1, embedding_dim), requires_grad=True) / math.sqrt(embedding_dim))
        self.C_init = Parameter(torch.normal(0, 1, (1, embedding_dim), requires_grad=True) / math.sqrt(embedding_dim))

        layer_dims = [embedding_dim] * (mlp_n_layers + 1)
        layer_activations = ([mlp_activation] * (mlp_n_layers - 1)) + [nn.Identity()] 
        self.LC_msg = MultiLayerPerceptron(layer_dims, layer_activations, p_dropout=config.model.mlp_dropout)
        self.CL_msg = MultiLayerPerceptron(layer_dims, layer_activations, p_dropout=config.model.mlp_dropout)

        self.L_update = LayerNormLSTMCell(input_size=2*embedding_dim, hidden_size=embedding_dim, 
                activation=lstm_activation, p_dropout=config.model.lstm_dropout)
        self.C_update = LayerNormLSTMCell(input_size=embedding_dim, hidden_size=embedding_dim, 
                activation=lstm_activation, p_dropout=config.model.lstm_dropout)

        self.L_attention = CausalSelfAttention(config.model.attention_nheads, 2*embedding_dim, 
                attn_pdrop=config.model.attention_pdrop)
        self.L_assignments = MultiLayerPerceptron(
                [2*embedding_dim, embedding_dim, 1], [mlp_activation, activations['sigmoid']], 
                p_dropout=config.model.mlp_dropout)
        self.L_vote = MultiLayerPerceptron(
                [2*embedding_dim, embedding_dim, 1], [mlp_activation, activations['sigmoid']], 
                p_dropout=config.model.mlp_dropout)


    def forward(self, adjacency_matrices: Tensor, batch_lit_counts: Tensor, device=torch.device("cpu")):
        self.device = device
        # adjacency_matrices: B x L x C
        B, L, C = adjacency_matrices.size()
        L_embeddings = self.L_init.repeat((B, L, 1)) # B x L x D
        C_embeddings = self.C_init.repeat((B, C, 1)) # B x C x D

        L_state_init = (L_embeddings, torch.zeros_like(L_embeddings).to(device)) 
        C_state_init = (C_embeddings, torch.zeros_like(C_embeddings).to(device))

        flip = self.batched_flip(batch_lit_counts).to(device)
        
        L_state_final, C_state_final = self.iterate(adjacency_matrices, L_state_init, C_state_init, flip)
        
        lit_readouts = torch.cat((L_state_final[0], torch.matmul(flip, L_state_final[0])), dim=2) # B x L x 2D
        lit_readouts = self.pos_lit_select(lit_readouts, batch_lit_counts) # B x L//2 x 2D

        mask = self.literal_mask(batch_lit_counts // 2).to(device)
        lit_readouts = self.L_attention(lit_readouts, mask=mask) # B x L//2 x 2D
        
        assignments = []
        votes = []
        for idx, batch_lits in enumerate(batch_lit_counts.data):
            assignments.append(self.L_assignments(lit_readouts[idx][:batch_lits//2]).squeeze(1)) # (L//2, 2D) -> (L//2, 1)
            votes.append(torch.mean(self.L_vote(lit_readouts[idx][:batch_lits//2])))
        
        return torch.stack(votes).float(), assignments

    def iterate(self, adjacency_matrices, L_state_init, C_state_init, flip_mtrx):
        L_state = L_state_init
        C_state = C_state_init
        
        for itr in range(self.iterations):
            LC_messages = torch.matmul(adjacency_matrices.transpose(1,2), self.LC_msg(L_state[0]))
            _, C_state = self.C_update(input=LC_messages, state=C_state)

            CL_messages = torch.matmul(adjacency_matrices, self.CL_msg(C_state[0])) 
            _, L_state = self.L_update(input=torch.cat((CL_messages, torch.matmul(flip_mtrx, L_state[0])), dim=2), 
                    state=L_state)
        return L_state, C_state

    def batched_flip(self, batch_counts) -> Tensor: 
        flip_matrices = []
        max_n = max(batch_counts)
        for batch_lits in batch_counts:
            n = batch_lits
            eye = torch.eye(n)
            flip_eye = torch.cat((eye[n//2:n], eye[0:n//2]))
            flip_eye = F.pad(flip_eye, (0, max_n - n, 0, max_n - n))
            flip_matrices.append(flip_eye)
        permute = torch.stack(flip_matrices)
        return permute
        
    def pos_lit_select(self, lit_embeddings, batch_counts) -> Tensor:
        B, L, D = lit_embeddings.size()
        lit_embeddings = lit_embeddings[:, :L//2, :]
        masks = []
        for b_n in batch_counts:
            mask = []
            mask.extend([True] * (b_n//2))
            mask.extend([False] * (L//2 - b_n//2))
            masks.append(torch.tensor(mask))
        masks = torch.stack(masks).unsqueeze(2).to(self.device)
        return torch.where(masks, lit_embeddings, torch.zeros_like(lit_embeddings))

    def literal_mask(self, batch_counts) -> Tensor:
        masks = []
        for b_n in batch_counts:
            mask = []
            mask.extend([1] * b_n)
            mask.extend([0] * (max(batch_counts) - b_n))
            mask = torch.tensor(mask)
            masks.append(mask)
        return torch.stack(masks).float()
            

class LayerNormLSTMCell(Module):

    def __init__(self, input_size:int, hidden_size: int, activation: Module=nn.Tanh(), p_dropout=0.0) -> None:
        super().__init__()
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.activation = activation

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=p_dropout)

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
        hy = self.dropout(hy)

        return hy, (hy, cy)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_head, embedding_dim, attn_pdrop=0.0):
        super().__init__()
        assert embedding_dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # regularization
        self.attn_drop = nn.Dropout(p=attn_pdrop)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            square_mask = torch.matmul(mask.view(B, -1, 1), mask.view(B, 1, -1))
            square_mask = square_mask.unsqueeze(dim=1)
            att = att.masked_fill(square_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(att.isnan(), float(0))

        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y # (B, T, C)


class NeuroSATLoss(Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relaxation_t = relaxation_types[config.general.relaxation]
        self.criterion = nn.BCELoss()

    def forward(self, assignments, formulas, device):
        satisfactions = []
        for var_assign, formula in zip(assignments, formulas):
            satisfaction = relax_sympy(formula, self.relaxation_t, var_assign, device)
            satisfactions.append(satisfaction)
        loss = self.criterion(torch.stack(satisfactions), torch.ones(len(assignments)).to(device))
        return loss


if __name__ == "__main__":
    with open("config.yaml") as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    dataset = NeuroSATDataset(root_dir="datasets/development", partition="train")

    adjacencies, literals, formulas, sats = collate_adjacencies([dataset[3], dataset[5]])
    print(adjacencies.shape, literals)
    adjacencies.requires_grad = True
    
    model = NeuroSATAssign(config)
    print(model)
    print(model.parameters())
    votes, assignments = model(adjacencies, literals)
    print(votes, assignments)
    print(assignments[0].shape, assignments[1].shape)
    (assignments[0].mean() + assignments[1].mean()).backward()
    print(adjacencies.grad)

