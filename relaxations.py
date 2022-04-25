import torch

class FuzzyRelaxation(object):

    @staticmethod
    def conjunction(a, b):
        raise NotImplementedError()
    
    @staticmethod
    def disjunction(a, b):
        raise NotImplementedError()

    @staticmethod
    def negation(a):
        return torch.sub(1, a)

    @staticmethod
    def implication(a, b):
        raise NotImplementedError()


class SGodel(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b):
        return torch.min(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def disjunction(a, b):
        return torch.max(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def negation(a):
        return FuzzyRelaxation.negation(a)

    @staticmethod
    def implication(a, b):
        return SGodel.disjunction(SGodel.negation(a), b)


class RGodel(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b):
        return torch.min(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def disjunction(a, b):
        return torch.max(torch.cat((a.view(1), b.view(1))))

    @staticmethod
    def negation(a):
        return FuzzyRelaxation.negation(a)

    @staticmethod
    def implication(a, b):
        return torch.where(a <= b, torch.tensor(1, dtype=torch.float32), b)      


class SProduct(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b):
        return torch.mul(a, b)

    @staticmethod
    def disjunction(a, b):
        return torch.sub(torch.add(a, b), torch.mul(a, b))

    @staticmethod
    def negation(a):
        return FuzzyRelaxation.negation(a)

    @staticmethod
    def implication(a, b):
        return SProduct.disjunction(SProduct.negation(a), b)


class RProduct(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b):
        return torch.mul(a, b)

    @staticmethod
    def disjunction( a, b):
        return torch.sub(torch.add(a, b), torch.mul(a, b))

    @staticmethod
    def negation(a):
        return FuzzyRelaxation.negation(a)

    @staticmethod
    def implication(a, b):
        return torch.where(a <= b, torch.tensor(1, dtype=torch.float32), torch.div(b, a))


class Lukasiewicz(FuzzyRelaxation):

    @staticmethod
    def conjunction(a, b):
        return torch.max(torch.cat((torch.tensor([0]), (a + b - 1).view(1))))

    @staticmethod
    def disjunction(a, b):
        return torch.min(torch.cat((torch.tensor([1]), (a + b).view(1))))

    @staticmethod
    def negation(a):
        return FuzzyRelaxation.negation(a)

    @staticmethod
    def implication(a, b):
        return Lukasiewicz.disjunction(Lukasiewicz.negation(a), b)


relaxation_types = {
    "godel": SGodel, "s_godel" : SGodel, "r_godel" : RGodel, 
    "product": SProduct, "s_product" : SProduct, "r_product" : RProduct, 
    "lukasiewicz" : Lukasiewicz
}

if __name__ == "__main__":
    pass
