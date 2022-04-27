import torch

class FuzzyRelaxation(object):

    @staticmethod
    def conjunction(a, b, device):
        raise NotImplementedError()
    
    @staticmethod
    def disjunction(a, b, device):
        raise NotImplementedError()

    @staticmethod
    def negation(a, device):
        return torch.sub(1, a)

    @staticmethod
    def implication(a, b, device):
        raise NotImplementedError()


class SGodel(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b, device):
        return torch.min(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def disjunction(a, b, device):
        return torch.max(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return SGodel.disjunction(SGodel.negation(a, device), b, device)


class RGodel(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b, device):
        return torch.min(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def disjunction(a, b, device):
        return torch.max(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return torch.where(a <= b, torch.tensor(1, dtype=torch.float32, device=device), b)      


class SProduct(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b, device):
        return torch.mul(a, b)

    @staticmethod
    def disjunction(a, b, device):
        return torch.sub(torch.add(a, b), torch.mul(a, b))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return SProduct.disjunction(SProduct.negation(a, device), b, device)


class RProduct(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b, device):
        return torch.mul(a, b)

    @staticmethod
    def disjunction(a, b, device):
        return torch.sub(torch.add(a, b), torch.mul(a, b))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return torch.where(a <= b, torch.tensor(1, dtype=torch.float32, device=device), torch.div(b, a))


class Lukasiewicz(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b, device):
        return torch.max(torch.cat((torch.tensor([0], device=device), (a + b - 1).view(1))))

    @staticmethod
    def disjunction(a, b, device):
        return torch.min(torch.cat((torch.tensor([1], device=device), (a + b).view(1))))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return Lukasiewicz.disjunction(Lukasiewicz.negation(a, device), b, device)

class Yager(FuzzyRelaxation):
    
    p = 1.5

    @staticmethod
    def conjunction(a, b, device):
        p = Yager.p
        return torch.max(torch.cat((torch.tensor([0], device=device), (1 - ((1 - a)**p + (1 - b)**p)**(1/p)).view(1))))

    @staticmethod
    def disjunction(a, b, device):
        p = Yager.p
        return torch.min(torch.cat((torch.tensor([1], device=device), ((a**p + b**p)**(1/p)).view(1))))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return Yager.disjunction(Yager.negation(a, device), b, device)


class MaxSATApprox(FuzzyRelaxation):
    
    @staticmethod
    def conjunction(a, b, device):
        return a + b

    @staticmethod
    def disjunction(a, b, device):
        return torch.min(torch.cat((torch.tensor([1], device=device), (a + b).view(1))))

    @staticmethod
    def negation(a, device):
        return FuzzyRelaxation.negation(a, device)

    @staticmethod
    def implication(a, b, device):
        return Lukasiewicz.disjunction(Lukasiewicz.negation(a, device), b, device)




relaxation_types = {
    "godel": SGodel, "s_godel" : SGodel, "r_godel" : RGodel, 
    "product": SProduct, "s_product" : SProduct, "r_product" : RProduct, 
    "lukasiewicz" : Lukasiewicz, "yager" : Yager, "maxsat": MaxSATApprox
}

if __name__ == "__main__":
    pass
