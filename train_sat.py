import argparse
import csv
import yaml
import json
import os
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from dataset import NeuroSATDataset, collate_adjacencies
from torch.utils.data import DataLoader
from torch import nn
from models import NeuroSATAssign, NeuroSATLoss
from utilities import compute_acc
from discriminators import discriminator_types
from relaxations import relaxation_types

def append_dict_to_csv(results_dict, csv_path, sep=","):
    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=list(results_dict.keys()))
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()    
        writer.writerow(results_dict)
    

def train_step(step_no: int, data_in: any, model: nn.Module, optimizers: any, criterions: any, config, 
    device=torch.device("cpu")):

    optimizer, scaler = optimizers
    sat_criterion, assignment_criterion = criterions

    optimizer.zero_grad()
    adj_matrices, lit_counts, formulas, sats = data_in
    adj_matrices, lit_counts, sats = adj_matrices.to(device), lit_counts.to(device), sats.to(device)
        
    with torch.cuda.amp.autocast():
        votes, assignments = model(adj_matrices, lit_counts, device=device)

    sat_loss = sat_criterion(votes, sats)
    assignment_loss = assignment_criterion(assignments, formulas, sats, device)
        
    loss = config.training.sat_loss_a * sat_loss + config.training.assn_loss_a * assignment_loss
        
    if not config.general.run_eval_mode:
        scaler.scale(loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        
    # TODO assure that this works correctly 
    correct_votes, correct_assignments = compute_acc(votes, sats, assignments, formulas, device)
        
    results = {
        'step': step_no,
        'sat accuracy': (correct_votes / len(sats)).item(), 
        'assignment precision': (correct_assignments / torch.sum(sats)).item(),
        'sat loss': sat_loss.item(),
        'assignment loss': assignment_loss.item()
    }
    return step_no + 1, results



def run_epoch(step_no, dataloader, model, optimizers, criterions, config, device):
    scheduler, optimizers = optimizers

    sat_accuracies = []
    assignment_precisions = []
    sat_losses = []
    assignment_losses = []
    csv_path = os.path.join("logs", "{}_running_results.csv".format(config.logging.experiment_tag))

    with tqdm(dataloader) as pbar:
        model = model.to(device)
        if not config.general.run_eval_mode: # train mode
            model.train()
            for batch in pbar:
                step_no, results = train_step(step_no, batch, model, optimizers, criterions, config, device)
                scheduler.step()
                pbar.set_postfix(results)
                append_dict_to_csv(results, csv_path)
                sat_accuracies.append(results['sat accuracy'])
                assignment_precisions.append(results['assignment precision'])
                sat_losses.append(results['sat loss'])
                assignment_losses.append(results['assignment loss'])
        else: # eval mode
            model.eval()
            with torch.no_grad():
                for batch in pbar:
                    step_no, results = train_step(step_no, batch, model, optimizers, criterions, config, device)
                    pbar.set_postfix(results)
                    append_dict_to_csv(results, csv_path)
                    sat_accuracies.append(results['sat accuracy'])
                    assignment_precisions.append(results['assignment precision'])
                    sat_losses.append(results['sat loss'])
                    assignment_losses.append(results['assignment loss'])

    epoch_results = {
        'epoch sat acc': np.mean(sat_accuracies),
        'epoch assignment precision': np.mean(assignment_precisions),
        'epoch sat loss': np.mean(sat_losses),
        'epoch assignment loss': np.mean(assignment_losses)
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
    model = NeuroSATAssign(config)

    #assert config.training.optimizer.step_rule == "AdamW", "AdamW is currently the only optimizer supported"
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.training.optimizer.learning_rate, 
            weight_decay=config.training.optimizer.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.training.optimizer.learning_rate,
            total_steps=config.training.optimizer.scheduler_steps)
    optimizers = (scheduler, (optimizer, scaler))

    sat_criterion = nn.BCELoss()
    assignment_criterion = NeuroSATLoss(config)
    criterions = (sat_criterion, assignment_criterion)

    step_no = 0
    
    batch_csv_path = os.path.join("logs", "{}_running_results.csv".format(config.logging.experiment_tag))
    epoch_csv_path = os.path.join("logs", "{}_epoch_results.csv".format(config.logging.experiment_tag))
    with open(batch_csv_path, 'w+') as f:
        pass
    with open(epoch_csv_path, 'w+') as f:
        pass

    for epoch in tqdm(range(config.training.epochs), desc='Epoch Loop'):
        step_no, epoch_results = run_epoch(step_no, dataloader, model, optimizers, criterions, config, device)
        append_dict_to_csv(epoch_results, epoch_csv_path)
        
