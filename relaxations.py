import torch
from torch import Tensor

def softmin(inputs: Tensor) -> Tensor:
    return torch.exp(-1 * inputs) / torch.sum(torch.exp(-1 * inputs))

class FuzzyAggregator(object):

    def __init__(self, device) -> None:
        self.device = device

    def universal(self, inputs: Tensor) -> Tensor:
        ''' Universal aggregator, i.e. "for all" in a Boolean formula. Analogous to n-ary conjunction. '''
        raise NotImplementedError()
    
    def existential(self, inputs: Tensor) -> Tensor:
        ''' Existential aggregator, i.e. "exists" in a Boolean formula. Analogous to n-ary disjunction. '''
        raise NotImplementedError()

class Godel(FuzzyAggregator):

    def __init__(self, device, soft=False) -> None:
        super().__init__(device)
        self.soft = soft

    def universal(self, inputs):
        if self.soft:
            return torch.sum(softmin(inputs) * inputs)
        return torch.min(inputs)

    def existential(self, inputs):
        if self.soft:
            return torch.sum(torch.softmax(inputs, dim=0) * inputs)
        return torch.max(inputs)

class Product(FuzzyAggregator):

    def universal(self, inputs):
        ''' Universal aggregator, i.e. "for all" in a Boolean formula. Analogous to n-ary conjunction. 
            This is the relaxed form, where outputs can be negative.
        '''
        return 1 + torch.sum(torch.log(inputs))

    def existential(self, inputs):
        return 1 - torch.prod(1 - inputs)

class Lukasiewicz(FuzzyAggregator):

    def universal(self, inputs):
        ''' Universal aggregator, i.e. "for all" in a Boolean formula. Analogous to n-ary conjunction. 
            This is the relaxed form, where outputs can be negative. Used in Hinge-Loss Markov Random Fields Paper
        '''
        return 1 - (len(inputs) - torch.sum(inputs))
        
    def existential(self, inputs):
        return torch.minimum(torch.sum(inputs), torch.tensor(1, device=self.device))

class Yager(FuzzyAggregator):
    
    def __init__(self, device, p=1) -> None:
        super().__init__(device)
        self.p = p

    def universal(self, inputs):
        ''' Universal aggregator, i.e. "for all" in a Boolean formula. Analogous to n-ary conjunction. 
            This is the relaxed form, where outputs can be negative.
        '''
        return 1 - len(inputs) - torch.linalg.norm(inputs, ord=self.p)

    def existential(self, inputs):
        return torch.minimum(torch.linalg.norm(inputs, ord=self.p), torch.tensor(1, device=self.device))

def relaxations(config):
    device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")
    relaxation_types = {
        "godel": Godel(device, soft=config.relaxations.godel_soft), "product": Product(device), 
        "lukasiewicz" : Lukasiewicz(device), "yager" : Yager(device, p=config.relaxations.yager_p)
    }
    return relaxation_types[config.relaxations.relaxation]


if __name__ == "__main__":
    pass
