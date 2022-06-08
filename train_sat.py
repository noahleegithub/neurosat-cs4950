import argparse
import yaml
import json
import os
from types import SimpleNamespace
from sat_models import MaxSATLoss
from tqdm import tqdm
import torch
import numpy as np

from dataset import NeuroSATDataset, collate_adjacencies
from torch.utils.data import DataLoader
from torch import nn
from sat_models import NeuroMaxSAT
from utilities import append_dict_to_csv, clause_accuracy
from relaxations import relaxations

def train_step(step_no: int, data_in: any, model: nn.Module, optimizer: any, criterion: any, config, device=torch.device("cpu")):

    optimizer.zero_grad()
    adj_matrices, batch_counts, formulas, sats = data_in
    adj_matrices, sats = adj_matrices.to(device), sats.to(device)

    with torch.cuda.amp.autocast():
        assignments, _ = model(adj_matrices, batch_counts) # (B, L, L) -> (B, V, 1) -> (B, V)
        assignments = assignments.squeeze()

        loss = criterion(assignments, formulas)
        
        if not config.general.run_eval_mode:
            assert loss.requires_grad, (loss, assignments)
            loss.backward()
            if config.training.optimizer.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
            optimizer.step()    
    

        batch_acc = clause_accuracy(assignments, formulas, device)

    results = {
        'step': step_no,
        'accuracy': batch_acc.item(), 
        'loss': loss.item(),
    }

    return step_no + 1, results



def run_epoch(step_no, dataloader, model, optimizer, criterion, config, device):

    accuracies = []
    losses = []
    csv_path = os.path.join("logs", "{}_running_results.csv".format(config.logging.experiment_tag))

    with tqdm(dataloader) as pbar:
        model = model.to(device)
        def run_batch(step_no):
            for batch in pbar:
                step_no, results = train_step(step_no, batch, model, optimizer, criterion, config, device)
                pbar.set_postfix(results)
                append_dict_to_csv(results, csv_path)
                accuracies.append(results['accuracy'])
                losses.append(results['loss'])
            return step_no

        if not config.general.run_eval_mode: # train mode
            model.train()
            step_no = run_batch(step_no)
        else: # eval mode
            model.eval()
            with torch.no_grad():
                step_no = run_batch(step_no)

    epoch_results = {
        'epoch acc': np.mean(accuracies),
        'epoch loss': np.mean(losses),
    }
    return step_no, epoch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to configuration file.")
    args = parser.parse_args()
    with open(args.config_path) as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    device = torch.device("cuda" if config.training.use_cuda and torch.cuda.is_available() else "cpu")
   
    # create datasets
    dataset = NeuroSATDataset(root_dir=config.general.dataset_root, partition=config.general.partition)

    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle,
            num_workers=config.training.num_workers, collate_fn=collate_adjacencies)

    # init model
    model = NeuroMaxSAT(config)

    #assert config.training.optimizer.step_rule == "AdamW", "AdamW is currently the only optimizer supported"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.optimizer.learning_rate, 
        weight_decay=config.training.optimizer.weight_decay)

    criterion = MaxSATLoss(config)

    step_no = 0

    if config.general.run_eval_mode:
        assert config.logging.load_checkpoint, "Need to use a trained model for evaluation"
    
    if config.logging.load_checkpoint:
        checkpoint_path = os.path.join("checkpoints", "{}.ckpt".format(config.logging.load_from_tag))
        assert os.path.isfile(checkpoint_path), "Checkpoint file does not exist"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if config.logging.resume_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step_no = checkpoint['step_no']

    batch_csv_path = os.path.join("logs", "{}_running_results.csv".format(config.logging.experiment_tag))
    epoch_csv_path = os.path.join("logs", "{}_epoch_results.csv".format(config.logging.experiment_tag))
    with open(batch_csv_path, 'w+') as f:
        pass
    with open(epoch_csv_path, 'w+') as f:
        pass

    checkpoint_path = os.path.join("checkpoints", "{}.ckpt".format(config.logging.experiment_tag))
    with tqdm(range(config.training.epochs), desc='Epoch Loop') as pbar:
        for epoch in pbar:
            step_no, epoch_results = run_epoch(step_no, dataloader, model, optimizer, criterion, config, device)
            pbar.set_postfix(epoch_results)
            append_dict_to_csv(epoch_results, epoch_csv_path)
            if config.logging.save_checkpoint:
                torch.save({
                    'step_no': step_no,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': epoch_results,
                }, checkpoint_path)
        
