import math
import yaml
import json
from types import SimpleNamespace

import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F

from models import activations, MultiLayerPerceptron, LayerNormLSTMCell, LayerNormGRUCell
from utilities import max_sat
from relaxations import relaxations
from dataset import NeuroSATDataset, collate_adjacencies
from ranking_models import DirectRanker

class NeuroMaxSAT(Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")
        embedding_dim = config.maxsat_model.hidden_dim
        mlp_n_layers = config.maxsat_model.mlp_layers
        mlp_activation = activations(config, config.maxsat_model.mlp_activation)
        lstm_activation = activations(config, config.maxsat_model.rnn_activation)

        self.iterations = config.maxsat_model.rnn_iters

        self.L_pos_init = Parameter(torch.normal(0, 1, (1, embedding_dim)) / math.sqrt(embedding_dim))
        self.L_neg_init = Parameter(torch.normal(0, 1, (1, embedding_dim)) / math.sqrt(embedding_dim))
        self.C_init = Parameter(torch.normal(0, 1, (1, embedding_dim)) / math.sqrt(embedding_dim))

        layer_dims = [embedding_dim] * (mlp_n_layers + 1)
        layer_activations = [mlp_activation] * mlp_n_layers
        self.LC_msg = MultiLayerPerceptron(layer_dims, layer_activations, p_dropout=config.maxsat_model.p_dropout,
                xavier_init=config.maxsat_model.xavier_init)
        self.CL_msg = MultiLayerPerceptron(layer_dims, layer_activations, p_dropout=config.maxsat_model.p_dropout,
                xavier_init=config.maxsat_model.xavier_init)

        self.L_update = LayerNormLSTMCell(input_size=2*embedding_dim, hidden_size=embedding_dim, 
                activation=lstm_activation, p_dropout=config.maxsat_model.p_dropout, forget_bias=config.maxsat_model.forget_bias)
        self.C_update = LayerNormLSTMCell(input_size=embedding_dim, hidden_size=embedding_dim, 
                activation=lstm_activation, p_dropout=config.maxsat_model.p_dropout, forget_bias=config.maxsat_model.forget_bias)

        self.L_assignments = DirectRanker(config, in_dim=embedding_dim)


    def forward(self, adjacency_matrices: Tensor, batch_lit_counts: Tensor):

        # adjacency_matrices: B x L x C
        B, L, C = adjacency_matrices.size()
        L_embeddings = torch.cat((self.L_pos_init.expand((B, L // 2, -1)), self.L_neg_init.expand((B, L // 2, -1))), dim=1) # B x L x D
        C_embeddings = self.C_init.expand((B, C, -1)) # B x C x D

        L_state_init = (L_embeddings, torch.zeros_like(L_embeddings).to(self.device)) 
        C_state_init = (C_embeddings, torch.zeros_like(C_embeddings).to(self.device))

        flip_matrix = self.batched_flip(batch_lit_counts).to(self.device)
        
        L_state_final, C_state_final = self.iterate(adjacency_matrices, L_state_init, C_state_init, flip_matrix)

        lit_readouts = L_state_final[0] # (B, L, D)

        all_lits_mask = NeuroMaxSAT.literals_mask(batch_lit_counts).to(self.device) # (B, L, 1)
        pos_lits_mask = NeuroMaxSAT.literals_mask((batch_lit_counts / 2).int()).to(self.device) # (B, L//2, 1)
        pos_lits_mask = F.pad(pos_lits_mask, (0,0,0,L//2))
        neg_lits_mask = torch.where(pos_lits_mask != all_lits_mask, 1, 0) # (B, L, 1)
    
        pos_lits = (lit_readouts * pos_lits_mask)[:,:L//2] # (B, L//2, D)
        neg_lits = (lit_readouts * neg_lits_mask)[:,:L//2] # (B, L//2, D)
        lit_assignments = torch.stack((pos_lits, neg_lits), dim=2) # (B, L//2, 2, D)

        lit_diffs, pos_vals, neg_vals = self.L_assignments(lit_assignments) # (B, L//2, 1)
    
        lit_diffs = lit_diffs * pos_lits_mask[:,:L//2]
        pos_vals = pos_vals * pos_lits_mask[:,:L//2]

        return lit_diffs, pos_vals

    def iterate(self, adjacency_matrices, L_state_init, C_state_init, flip_mtrx):
        L_hidden, L_cell = L_state_init
        C_hidden, C_cell = C_state_init
        
        for _ in range(self.iterations):
            LC_messages = torch.matmul(adjacency_matrices.transpose(1,2), self.LC_msg(L_hidden))
            _, C_state = self.C_update(input=LC_messages, state=(C_hidden.detach(), C_cell.detach()))
            C_hidden, C_cell = C_state

            CL_messages = torch.matmul(adjacency_matrices, self.CL_msg(C_hidden)) 
            _, L_state = self.L_update(input=torch.cat((CL_messages, torch.matmul(flip_mtrx, L_hidden)), dim=2), 
                    state=(L_hidden.detach(), L_cell.detach()))
            L_hidden, L_cell = L_state
    
        return L_state, C_state

    @staticmethod
    def batched_flip(batch_counts) -> Tensor: 
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

    @staticmethod
    def literals_mask(batch_counts) -> Tensor:
        B, L = len(batch_counts), max(batch_counts)
        indices = torch.arange(L).expand((B,-1))
        mask = torch.where(indices < batch_counts.unsqueeze(1), 1, 0)
        mask = mask.unsqueeze(2)
        return mask # (B, L, 1)

class MaxSATLoss(Module):
    
    def __init__(self, config):
        super().__init__()
        self.relaxation = relaxations(config)
        self.device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")

    def forward(self, assignments, formulas):
        assignments = torch.sigmoid(assignments)
        sats = torch.stack([max_sat(a, f, self.relaxation, self.device) for a, f in zip(assignments, formulas)])
        loss = 1 - sats
        assert torch.all(loss >= 0), loss
        return torch.mean(loss)

if __name__ == "__main__":
    with open("configurations/config_sat.yaml") as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    dataset = NeuroSATDataset(root_dir="datasets/development", partition="train")

    adjacencies, literals, formulas, sats = collate_adjacencies([dataset[3], dataset[5]])

    adjacencies = adjacencies.to(torch.device("cuda"))
    print(adjacencies.shape, literals)
    
    model = NeuroMaxSAT(config).to(torch.device("cuda"))
    print(model)
    optim = torch.optim.Adam(model.parameters())
    optim.zero_grad()

    assignments = model(adjacencies, literals)
    print(assignments)
    print(assignments[0].shape, assignments[1].shape)
    torch.sum(assignments).backward()

    print(model.L_pos_init.grad)
    print(adjacencies.grad)
    optim.step()
    
    #print(model.L_assignments(torch.zeros(256)))

