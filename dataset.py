import os
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
    return torch.stack(padded_matrices), torch.tensor(num_literals), formulas, torch.tensor(sats).float()



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
