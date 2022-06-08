import yaml
import json
from typing import Sequence
from types import SimpleNamespace
from itertools import combinations, permutations

from sympy import symbols, Symbol
from sympy.logic.boolalg import BooleanFunction, And, Or, Not
import numpy as np
import torch
from torch import nn, Tensor

from dataset import collate_adjacencies
from models import MultiLayerPerceptron, activations
from utilities import combinations_2, sympy_to_weighted_adjacency, ndarray_to_tuples


class DirectRanker(nn.Module):

    def __init__(self, config: SimpleNamespace, in_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim = config.rank_model.input_dim
        hidden_dim = config.rank_model.hidden_dim
        n_layers = config.rank_model.mlp_layers
        p_drop=config.rank_model.p_dropout
        xavier_init=config.rank_model.xavier_init
        mlp_activation = activations(config, config.rank_model.mlp_activation)

        self.enc_ffnn = MultiLayerPerceptron(layer_dims=[in_dim]+[hidden_dim]*n_layers,
            layer_activations=[mlp_activation]*n_layers, 
            p_dropout=p_drop, xavier_init=xavier_init)
        self.out_ffnn = MultiLayerPerceptron(layer_dims=[hidden_dim, 1], layer_activations=[nn.Identity()],
            p_dropout=p_drop, bias=False, xavier_init=xavier_init)

    def forward(self, x: Tensor):
        N = x.shape[-2]
        assert N == 2
        x1 = x[..., 0, :]
        x2 = x[..., 1, :]

        x1 = self.enc_ffnn(x1)
        x2 = self.enc_ffnn(x2)

        return self.out_ffnn(x1 - x2), self.out_ffnn(x1), self.out_ffnn(x2)


class MaxSATRanker(nn.Module):

    def __init__(self, config: SimpleNamespace, pair_ranker: nn.Module, maxsat_solver: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.pair_ranker = pair_ranker
        self.maxsat_solver = maxsat_solver
        self.device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")

    def forward(self, features):
        B, N, D = features.shape

        pairwise_indices = combinations_2(np.arange(N), batched=False) # (NC2, 2)
        variables = symbols("v1:{}".format(len(pairwise_indices)+1))
        pair2idx = {pair : idx for idx, pair in enumerate(ndarray_to_tuples(pairwise_indices))}

        pairwise_features = combinations_2(features) # (B, NC2, 2, D)

        ranker_predictions = self.pair_ranker(pairwise_features).squeeze() # (B, NC2, 2, D) -> (B, NC2, 1) -> (B, NC2)

        formulas = self.create_cnfs(B, N, ranker_predictions, pair2idx, variables)
        variable_probabilities = torch.sigmoid(ranker_predictions) # (B, NC2)
        adj_matrices = [sympy_to_weighted_adjacency(cnf, weights, device=self.device) for cnf, weights in zip(formulas, variable_probabilities)]
        batch_counts = torch.tensor([N] * B, device=self.device)

        adj_matrices, batch_counts, formulas, _ = collate_adjacencies(zip(adj_matrices, formulas, [0] * B))
        
        maxsat_assignments = self.maxsat_solver(adj_matrices, batch_counts).squeeze() # (B, NC2 * 2, C) -> (B, NC2, 1) -> (B, NC2)
        return ranker_predictions, maxsat_assignments, formulas

    def create_cnfs(self, batches: int, n_docs: int, pair_predictions: Tensor, pair2idx: dict, variables: Sequence[Symbol]):
        B, N = batches, n_docs
        cnfs = []
        for sample in range(B):
            clauses = []
            for i, (x, y, z) in enumerate(combinations(np.arange(N), 3)):
                sign_xy = torch.sign(pair_predictions[sample, pair2idx[(x, y)]])
                sign_yz = torch.sign(pair_predictions[sample, pair2idx[(y, z)]])
                if sign_xy == sign_yz:
                    clauses.append(Or(
                        Not(variables[pair2idx[(x, y)]]), 
                        Not(variables[pair2idx[(y, z)]]), 
                        variables[pair2idx[(x,z)]]
                    ))
            cnfs.append(And(*clauses))
        return cnfs
        

if __name__ == "__main__":
    with open("configurations/config_rank_sat.yaml") as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    print(config)
    model = DirectRanker(config)
    print(model)