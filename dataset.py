import os
import numpy as np
import torch
from typing import Sequence, Tuple
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from sympy.logic.boolalg import BooleanFunction
from sympy.logic.utilities import dimacs

from utilities import dimacs_to_adjacency, direct_sum, pad_sparse_matrix
from discriminators import discriminator_types
from relaxations import relaxation_types

class NeuroSATDataset(Dataset):
    
    def __init__(self, root_dir: str, partition: str) -> None:
        super().__init__()
        if partition not in ["train", "validation", "test"]:
            raise ValueError()

        self.root_dir = os.path.join(root_dir, partition)
        self.files = os.listdir(self.root_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> Tuple[Tensor, Module]:
        dimacs_path = os.path.join(self.root_dir, self.files[index])
        sat = int(dimacs_path[dimacs_path.index("sat=") + 4])
        boolean_formula = dimacs.load_file(dimacs_path)
        return dimacs_to_adjacency(dimacs_path, sparse=False), boolean_formula, sat

def collate_adjacencies(batch: Sequence[Tuple[Tensor, BooleanFunction, int]]) -> Tuple[Tensor, Sequence[int], Sequence[BooleanFunction], Tensor]:
    adjacency_matrices, formulas, sats = zip(*batch)
    num_literals = [m.shape[0] for m in adjacency_matrices]
    num_clauses = [m.shape[1] for m in adjacency_matrices]
    max_l = max(num_literals)
    max_c = max(num_clauses)
    padded_matrices = []
    for m in adjacency_matrices:
        if m.is_sparse:
            padded_matrices.append(pad_sparse_matrix(m, max_l, max_c))
        else:
            padded_matrices.append(F.pad(m, (0, max_c - m.shape[1], 0, max_l - m.shape[0])))
    return torch.stack(padded_matrices), torch.tensor(num_literals).int(), formulas, torch.tensor(sats).float()

class MSLR10KDataset(Dataset):
    
    def __init__(self, root_dir, folds, partition: str, seed=None, mode="list", k=10):
        super().__init__()
        if len(folds) == 0 or partition not in ["train", "vali", "test"]:
            raise ValueError()
        self.queries = {}
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.d = 136
        self.k = k

        for fold in folds:
            file_path = os.path.join(root_dir, fold, partition + ".txt")
            with open(file_path) as f:
                for line in f.readlines():
                    label, line = line.split(' ', 1)
                    label = int(label)

                    qid, line = line.split(' ', 1)
                    _, qid = qid.split(':')
                    qid = int(qid)

                    self.queries[qid] = self.queries.get(qid, {})
                    self.queries[qid][label] = self.queries[qid].get(label, [])

                    feature_vec = []
                    for feature in line.split():
                        idx, val = feature.split(':')
                        feature_vec.append(float(val))
                    self.queries[qid][label].append(feature_vec)
        for qid, query in dict(self.queries).items():
            if len(query.keys()) < 2:
                del self.queries[qid]
                continue
            for label in range(5):
                if label not in query:
                    continue
                self.queries[qid][label] = np.array(self.queries[qid][label])
        self.idx2qid = {idx: key for idx, key in enumerate(self.queries.keys())}
        

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[self.idx2qid[idx]]
        if self.mode == "list":
            sorted_results = np.zeros((self.k, self.d))
            sorted_relevances = np.sort(self.rng.choice(list(query.keys()), size=self.k))[::-1]
            for idx, relevance in enumerate(sorted_relevances):
                sorted_results[idx] = self.rng.choice(query[relevance])

            permuted_indices = self.rng.permutation(self.k)

            permuted_vectors = sorted_results[permuted_indices]
            permuted_vectors = (permuted_vectors - permuted_vectors.mean(axis=1, keepdims=True)) / permuted_vectors.std(axis=1, keepdims=True)
            return permuted_vectors, sorted_relevances[permuted_indices]
        elif self.mode == "pair":
            relevances = self.rng.choice(list(query.keys()), size=2, replace=False)
            pair = np.zeros((2, 136))
            pair[0] = self.rng.choice(query[relevances[0]])
            pair[1] = self.rng.choice(query[relevances[1]])
            pair = (pair - pair.mean(axis=1, keepdims=True)) / pair.std(axis=1, keepdims=True)
            return pair, 1. if relevances[0] > relevances[1] else 0.
        else:
            raise ValueError("not a valid return mode for MSLR10K")

if __name__ == "__main__":
    nsat = NeuroSATDataset("./datasets/development", "train")
    print(len(nsat))
    print(nsat[30])
    print(nsat[31])
    combined = collate_adjacencies([nsat[30], nsat[31]])
    print(combined[0])
    print(combined[1])
    print(combined[2])
    print(combined[3])

    mslr10k = MSLR10KDataset("./datasets/MSLR-WEB10K", ["Fold1"], "train", mode="list")
    print(len(mslr10k))
    print(mslr10k[0])

