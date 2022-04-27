import argparse
import yaml
import json
from types import SimpleNamespace
import torch
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import NeuroSATDataset, collate_adjacencies
from models import NeuroSATAssign

class CustomNeuroSATWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        adj, lits, device = batch
        votes, assignments = self.model(adj, lits, device)
        return votes

class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        adj, lits, formula, sats = batch_data
        return (adj, lits, torch.device("cuda")), sats



class CustomValIter(ValDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        adj, lits, formula, sats = batch_data
        return (adj, lits, torch.device("cuda")), sats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to configuration file.")
    args = parser.parse_args()
    with open(args.config_path) as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")

    # create datasets
    train_dataset = NeuroSATDataset(root_dir=config.general.dataset_root, partition="train")
    val_dataset = NeuroSATDataset(root_dir=config.general.dataset_root, partition="validation")

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle,
            num_workers=config.training.num_workers, collate_fn=collate_adjacencies)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle,
            num_workers=config.training.num_workers, collate_fn=collate_adjacencies)

    custom_train_iter = CustomTrainIter(train_dataloader)
    custom_val_iter = CustomValIter(val_dataloader)
    # init model
    model = CustomNeuroSATWrapper(NeuroSATAssign(config))
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)

    criterion = nn.BCELoss()
   
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(custom_train_iter, val_loader=custom_val_iter, end_lr=1e-2, num_iter=1000, step_mode="linear")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
