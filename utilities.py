import numpy as np
import torch
from torch import Tensor
from typing import Sequence, Tuple

def dimacs_to_adjacency(in_fname: str, sparse: bool=True):
    """ Reads the given file in DIMACS format into a sparse Tensor.
        The shape of the tensor is (2n, m), where n is the number of literals and m
        the number of clauses.
    """
    with open(in_fname, 'r') as f:
        _, form, num_lits, num_clauses = f.readline().split()
        if not form == 'cnf':
            raise ValueError(f"File {in_fname} not in DIMACS CNF format.")
        num_lits = int(num_lits)
        num_clauses = int(num_clauses)

        adj_arr = np.zeros((2 * num_lits, num_clauses))

        for clause_idx, clause in enumerate(f):
            lits = [int(x) for x in clause.split()[:-1]]
            lits_idxs = [(x - 1) if x > 0 else (-x + num_lits - 1) for x in lits]
            for l_idx in lits_idxs:
                adj_arr[l_idx, clause_idx] = 1
                
        adj_tensor = torch.tensor(adj_arr).float()
        if sparse:
            adj_tensor = adj_tensor.to_sparse()
        return adj_tensor

def direct_sum(matrices: Sequence[Tensor]) -> Tensor:
    row_indices = []
    col_indices = []
    cur_size = [0, 0]
    for matrix in matrices:
        if not matrix.is_sparse:
            matrix = matrix.to_sparse()
        indices = matrix._indices()
        new_row_indices = indices[0] + cur_size[0]
        new_col_indices = indices[1] + cur_size[1]
        row_indices.extend(new_row_indices)
        col_indices.extend(new_col_indices)
        cur_size[0] += matrix.shape[0]
        cur_size[1] += matrix.shape[1]
    return torch.sparse_coo_tensor(
            indices=[row_indices, col_indices],
            values=torch.ones((len(row_indices))),
            size=cur_size,
            dtype=torch.float32)


def compute_acc(predictions, lit_sizes, discriminators):
    lit_index = 0
    num_correct = 0
    for lits_size, discrim in zip(lit_sizes, discriminators):
        pred = predictions[0 + lit_index : lits_size + lit_index]
        lit_index += lits_size
        satisfies = discrim(pred)
        satisfies = 1 if satisfies > 0.5 else 0 
        num_correct += satisfies
    accuracy = num_correct / len(lit_sizes)
    return accuracy

        
if __name__ == "__main__":
    test_cnf1 = "./dataset/development/sr_n=0003_pk2=0.30_pg=0.40_t=0_sat=1.dimacs"
    test_cnf2 = "./dataset/development/sr_n=0003_pk2=0.30_pg=0.40_t=1_sat=1.dimacs"
    test_cnf3 = "./dataset/development/sr_n=0004_pk2=0.30_pg=0.40_t=3_sat=1.dimacs"
    
    test_adj1 = dimacs_to_adjacency(test_cnf1)
    test_adj2 = dimacs_to_adjacency(test_cnf2)
    test_adj3 = dimacs_to_adjacency(test_cnf3)
    
    torch.set_printoptions(profile='full', linewidth=1000)
    print(direct_sum([test_adj1, test_adj2, test_adj3]).to_dense())
