import os
import torch
from typing import Sequence, Tuple
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import Module

from utilities import dimacs_to_adjacency, direct_sum
from discriminators import discriminator_types
from relaxations import relaxation_types

class NeuroSATDataset(Dataset):
    
    def __init__(self, root_dir: str, partition: str, discriminator: str, relaxation: str) -> None:
        super().__init__()
        if partition not in ["train", "validation", "test"]:
            raise ValueError()
        if discriminator not in discriminator_types:
            raise ValueError()
       
        self.root_dir = os.path.join(root_dir, partition)
        self.files = os.listdir(self.root_dir)
        self.discriminator = discriminator_types[discriminator]
        self.relaxation = relaxation_types[relaxation]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> Tuple[Tensor, Module]:
        dimacs_path = os.path.join(self.root_dir, self.files[index])
        return dimacs_to_adjacency(dimacs_path), self.discriminator(dimacs_path, self.relaxation)

def collate_adjacencies(batch: Sequence[Tuple[Tensor, Module]]) -> Tuple[Tensor, Sequence[int], Sequence[Module]]:
    num_literals = [problem[0].shape[0] for problem in batch]
    discriminators = [problem[1] for problem in batch]
    return direct_sum([problem[0] for problem in batch]), num_literals, discriminators



if __name__ == "__main__":
    nsat = NeuroSATDataset("./dataset/development", "train", "minisat")
    print(len(nsat))
    print(nsat[0])
    print(nsat[1])
    combined = collate_adjacencies([nsat[0], nsat[1]])
    print(combined[0])
    print(combined[1])
