import csv
import os
import numpy as np
import torch
from torch import Tensor
from typing import Sequence
from sympy import Symbol
from sympy.logic.boolalg import BooleanFunction, Not
from sympy.logic.utilities import dimacs
from relaxations import FuzzyAggregator, Lukasiewicz

def append_dict_to_csv(results_dict, csv_path, sep=","):
    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=list(results_dict.keys()))
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()    
        writer.writerow(results_dict)

def dimacs_to_adjacency(in_fname: str, sparse: bool=True):
    """ Reads the given file in DIMACS format into a Tensor.
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

def pad_sparse_matrix(matrix: Tensor, new_w, new_h) -> Tensor:
    return torch.sparse_coo_tensor(
            indices=matrix.indices(),
            values=torch.ones(len(matrix.values())),
            size=(new_w, new_h)
        )

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

def extract_int_from_str(s):
    """ Get the integer that is present in the given string. 
    """
    try:
        return int(''.join(filter(str.isdigit, s)))
    except ValueError:
        raise ValueError("string did not contain an int.")

    
def clause_accuracy(assignments: Tensor, formulas: Sequence[BooleanFunction], device=torch.device("cpu")):
    relaxation = Lukasiewicz(device)
    assignments = torch.where(assignments > 0, 1., 0.)
    max_sats = torch.tensor([len(f.args) for f in formulas], device=device).float()
    sats = torch.stack([max_sat(a, f, relaxation, device) for a, f in zip(assignments, formulas)])
    accuracies = (max_sats - (1 - sats)) / max_sats
    return torch.mean(accuracies)

def max_sat(values: Tensor, cnf: BooleanFunction, relaxation: FuzzyAggregator, device=torch.device("cpu")):
    clause_satisfactions = []
    for clause in cnf.args:
        literals = clause.args
        if isinstance(clause, Symbol):
            literals = [clause]

        variables = torch.stack([values[extract_int_from_str(str(lit)) - 1] for lit in literals])
        negated = torch.tensor([isinstance(lit, Not) for lit in literals], device=device)
        variables = torch.where(negated, 1 - variables, variables)
        clause_satisfactions.append(relaxation.existential(variables))
    cnf_satisfaction = relaxation.universal(torch.stack(clause_satisfactions))
    return cnf_satisfaction

def ndcg_score(relevance_ranking: Tensor, optimal_ranking: Tensor=None, k: int=None, debug=False):
    if k is None or k > len(relevance_ranking):
        k = len(relevance_ranking)
    relevance_ranking = relevance_ranking[:k].cpu()

    if optimal_ranking is not None: optimal_ranking = optimal_ranking.cpu()
    else: optimal_ranking, _ = torch.sort(relevance_ranking, descending=True)
    optimal_ranking = optimal_ranking[:k]

    def dcg_score(relevance_ranking):
        scores = torch.div((2**relevance_ranking - 1), torch.log2(torch.arange(2, k+2).float()))
        return torch.sum(scores)

    dcg = dcg_score(relevance_ranking)
    idcg = dcg_score(optimal_ranking)
    if debug: print(dcg, idcg)
    return dcg / idcg # Can return NaN!

def average_precision(ranking: Tensor, k: int=None):
    ranking = ranking.cpu()
    if k is None or k > len(ranking):
        k = len(ranking)
    # ranking is binarized
    ranking = ranking[:k]
    gtp = torch.sum(ranking)
    
    p_at_k = torch.tril(ranking.view(1,-1).expand((k,-1)), diagonal=1).sum(axis=1) / torch.arange(1, k+1).float()
    return torch.sum(p_at_k * ranking) / gtp # Can return NaN!


def combinations_2(data: np.ndarray, batched=True):
    if batched:
        n = data.shape[1]
    else:
        n = data.shape[0]
    combinations = np.transpose(np.triu_indices(n, 1))
    if batched:
        return data[:, combinations]
    else:
        return data[combinations]

    
if __name__ == "__main__":
    test_cnf1 = "./tests/testformula.cnf"
    formula = dimacs.load_file(test_cnf1)
    print(formula)
    
    print(max_sat(formula, torch.tensor([.3,.2,.7,.8,.0]), torch.device("cpu")))

    truth = torch.tensor([3,3,3,2,2,2]).float()
    print(ndcg_score(torch.tensor([3,2,3,0,1,2]).float(), optimal_ranking=truth, debug=True))


