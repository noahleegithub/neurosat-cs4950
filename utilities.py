import numpy as np
import torch
from torch import Tensor
from typing import Sequence, Tuple
from sympy import Symbol
from sympy.logic.boolalg import BooleanFunction, And, Or, Not, Implies
from sympy.logic.utilities import dimacs
from relaxations import FuzzyRelaxation, relaxation_types

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


def compute_acc(votes, sats, assignments, formulas, device):
    relaxation = relaxation_types['product']
    votes = torch.where(votes > 0.5, 1., 0.)
    correct_sat_predictions = torch.sum(votes == sats)

    assignments = torch.where(assignments > 0.5, 1., 0.)
    satisfactions = torch.tensor([relax_sympy(f, relaxation, a, device) for a, f in zip(assignments, formulas)], 
            device=device)

    return correct_sat_predictions, torch.sum(satisfactions * sats) 

def extract_int_from_str(s):
    """ Get the integer that is present in the given string. 
    """
    try:
        return int(''.join(filter(str.isdigit, s)))
    except ValueError:
        raise ValueError("string did not contain an int.")

def relax_sympy(expr: BooleanFunction, relaxation_t: FuzzyRelaxation, values: torch.Tensor, device):
    """ Using the given relaxation of logic and SymPy expression, computes the value of the overall expression. 
    Args: 
        expr (BooleanFunction): A SymPy boolean expression in NNF form (And, Or, or Not).
            Not can only be applied to literals.
        relaxation_t (FuzzyRelaxation): Class with real-valued logic implementations of conjunction, 
            disjunction and negation.
        values (Tensor[float]): A tensor that contains floats 0 <= x <= 1. Length of values should at least
            be one greater than the max variable in expr.
    """
    if isinstance(expr, And):
        children = expr.args
        ret_value = torch.tensor(1.).to(device)
        for child in children:
            ret_value = relaxation_t.conjunction(ret_value, relax_sympy(child, relaxation_t, values, device), device)
        return ret_value
    elif isinstance(expr, Or):
        children = expr.args
        ret_value = torch.tensor(0.).to(device)
        for child in children:
            ret_value = relaxation_t.disjunction(ret_value, relax_sympy(child, relaxation_t, values, device), device)
        return ret_value
    elif isinstance(expr, Not):
        child = expr.args[0]
        ret_value = relaxation_t.negation(relax_sympy(child, relaxation_t, values, device), device)
        return ret_value
    elif isinstance(expr, Implies):
        children = expr.args
        ret_value = relaxation_t.implication(
            relax_sympy(children[0], relaxation_t, values, device), 
            relax_sympy(children[1], relaxation_t, values, device), device)
        return ret_value
    elif isinstance(expr, Symbol):
        symbol_num = extract_int_from_str(str(expr))
        ret_value = values[symbol_num - 1]
        return ret_value
    elif isinstance(expr, BooleanTrue):
        return torch.tensor(1.).to(device)
    else:
        raise ValueError("Expression is invalid")

def max_sat(cnf: BooleanFunction, values: Tensor, device):
    satisfaction = torch.tensor(0., device=device)
    for clause in cnf.args:
        literals = clause.args
        if isinstance(clause, Symbol):
            literals = [clause]

        variables = torch.tensor([values[extract_int_from_str(str(lit)) - 1] for lit in literals], device=device)
        negated = torch.tensor([True if isinstance(lit, Not) else False for lit in literals], device=device)
        variables = torch.where(negated, 1 - variables, variables)
        satisfaction += torch.minimum(torch.sum(variables), torch.tensor(1, device=device))
    return satisfaction

def ndcg_score(relevance_ranking: Sequence[float], optimal_ranking: Sequence[float]=None, k: int=None):
    if k is None or k > len(relevance_ranking):
        k = len(relevance_ranking)
    if optimal_ranking is None:
        optimal_ranking = np.sort(relevance_ranking)[::-1]

    def dcg_score(relevance_ranking):
        scores = np.array(relevance_ranking) / np.log2(np.arange(2, len(relevance_ranking) + 2))
        return np.sum(scores[:k])
    dcg = dcg_score(relevance_ranking)
    idcg = dcg_score(np.sort(relevance_ranking)[::-1])
    return dcg / idcg

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
    print(relax_sympy(formula, relaxation_types['godel'], torch.tensor([.3, .2, .7, .8, 0], dtype=float), torch.device("cpu")))
    
    print(max_sat(formula, torch.tensor([.3,.2,.7,.8,.0]), torch.device("cpu")))

    print(ndcg_score([1,1,0,0.5,0], k=3))
