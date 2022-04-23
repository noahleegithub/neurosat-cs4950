import torch
from PyMiniSolvers import minisolvers
from torch import Tensor
from torch.nn import Module
from relaxations import relaxation_types, FuzzyRelaxation

class Discriminator(Module):

    def __init__(self, dimacs_file: str, relaxation: FuzzyRelaxation)-> None:
        super().__init__()
        self.dimacs_file = dimacs_file
        self.relaxation = relaxation

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError()


class MiniSATOracle(Discriminator):

    def __init__(self, dimacs_file: str, relaxation: FuzzyRelaxation) -> None:
        super().__init__(dimacs_file, relaxation)
        self.solver = minisolvers.MinisatSolver()
        with open(dimacs_file, 'r') as f:
            _, form, num_lits, num_clauses = f.readline().split()
            if not form == 'cnf':
                raise ValueError(f"File {dimacs_file} not in DIMACS CNF format.")
            for l in range(int(num_lits)):
                self.solver.new_var()

            for clause in f:
                lits = [int(x) for x in clause.split()[:-1]]
                self.solver.add_clause(lits)
        
    def forward(self, input: Tensor) -> Tensor:
        input = torch.flatten(input)
        positive_indices = torch.flatten(torch.nonzero(input > 0.5)) + 1
        satisfied = self.solver.check_complete(positive_lits=positive_indices)
        if satisfied:
            return torch.tensor(1)
        else:
            return torch.tensor(0)


class FuzzyCNFDiscriminator(Discriminator):

    def __init__(self, dimacs_file: str, relaxation: FuzzyRelaxation) -> None:
        super().__init__(dimacs_file, relaxation)

        with open(dimacs_file, 'r') as f:
            _, form, num_lits, num_clauses = f.readline().split()
            if not form == 'cnf':
                raise ValueError(f"File {dimacs_file} not in DIMACS CNF format.")
            
            self.clauses = []

            for clause in f:
                lits = [int(x) for x in clause.split()[:-1]]
                self.clauses.append(lits)


    def forward(self, input: Tensor) -> Tensor:
        clause_values = []
        for clause in self.clauses:
            fuzzy_literals = []
            for lit in clause:
                if lit > 0:
                    fuzzy_literals.append(input[lit - 1])
                else:
                    fuzzy_literals.append(self.relaxation.negation(input[-1 * lit - 1]))
            
            clause_value = fuzzy_literals[0]
            for fuzzy_lit in fuzzy_literals[1:]:
                clause_value = self.relaxation.disjunction(clause_value, fuzzy_lit)
            clause_values.append(clause_value)

        formula_value = clause_values[0]
        for c_val in clause_values[1:]:
            formula_value = self.relaxation.conjunction(formula_value, c_val)
        return formula_value


class FuzzyDDNNFDiscriminator(Discriminator):

    def __init__(self, dimacs_file: str, relaxation: FuzzyRelaxation)-> None:
        super().__init__(dimacs_file, relaxation)
        

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError()



discriminator_types = {"minisat" : MiniSATOracle, "fuzzy_cnf" : FuzzyCNFDiscriminator}

if __name__ == "__main__":
    print(discriminator_types)
    print(relaxation_types)
    f_cnf = discriminator_types["fuzzy_cnf"]
    f_cnf = f_cnf("./tests/testformula.cnf", relaxation_types["r_product"])
    inp = torch.tensor([0.3504, 0.5041, 0.8576, 0.8738], requires_grad=True)
    result = f_cnf(inp)
    print(result)
    result.backward()
    print(inp.grad)

