import argparse
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

def append_df_to_csv(df, csvFilePath, sep=","):
    if os.path.exists(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)

def train_loop(config, dataloader, model, sat_criterion, assignment_criterion, optimizer, device, csv_path, epoch):
    model.train()
    model = model.to(device)
    
    n_correct_votes = 0
    n_correct_assignments = 0
    n_satisfiable_problems = 0

    sat_losses = []
    assignment_losses = []
    pbar = tqdm(dataloader)
    for it, batch in enumerate(pbar):
        optimizer.zero_grad()
        adj_matrices, lit_counts, formulas, sats = batch
        adj_matrices = adj_matrices.to(device)
        sats = sats.to(device)
  
        votes, assignments = model(adj_matrices, lit_counts, device=device)

        sat_loss = sat_criterion(votes, sats)
        assignment_loss = assignment_criterion(assignments, formulas, device)

        sat_loss.backward(retain_graph=True)
        assignment_loss.backward() # keeping the losses separate is supposed to be helpful for adam

        nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)

        optimizer.step()
        
        correct_votes, correct_assignments = compute_acc(votes, sats, assignments, formulas, device)
        n_correct_votes += correct_votes
        n_correct_assignments += correct_assignments
        n_satisfiable_problems += torch.sum(sats)

        sat_losses.append(sat_loss.item())
        assignment_losses.append(assignment_loss.item())

        batch_results = {
                'epoch': epoch,
                'batch iter': it,
                'sat acc': (correct_votes / len(sats)).item(), 
                'assignment prec': (correct_assignments / torch.sum(sats)).item(),
                'sat loss': sat_losses[-1],
                'assignment loss': assignment_losses[-1]
            }

        pbar.set_description("SAT Acc: {0:.3f} | Assign Prec: {1:.3f} | SAT Loss: {2:.3f} | Assign Loss: {3:.3f}".format(
            batch_results['sat acc'], batch_results['assignment prec'], batch_results['sat loss'], 
            batch_results['assignment loss']))
        batch_df = pd.DataFrame()
        batch_df.append(batch_results, ignore_index=True)
        append_df_to_csv(batch_df, csv_path)


    voting_acc = n_correct_votes / len(dataloader.dataset)
    assignment_precision = n_correct_assignments / n_satisfiable_problems
    voting_loss = np.sum(sat_losses) / len(dataloader.dataset)
    assignment_loss = np.sum(assignment_losses) / len(dataloader.dataset)
 
    return voting_acc, assignment_precision, voting_loss, assignment_loss

def val_loop(config, dataloader, model, sat_criterion, assignment_criterion, device):
    model.eval()
    model = model.to(device)
    
    n_correct_votes = 0
    n_correct_assignments = 0
    n_satisfiable_problems = 0

    sat_losses = []
    assignment_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Batch Loop'):
            optimizer.zero_grad()
            adj_matrices, lit_counts, formulas, sats = batch
            adj_matrices = adj_matrices.to(device)
            sats = sats.to(device)
  
            votes, assignments = model(adj_matrices, lit_counts, device=device)

            sat_loss = sat_criterion(votes, sats)
            assignment_loss = assignment_criterion(assignments, formulas, device)

            correct_votes, correct_assignments = compute_acc(votes, sats, assignments, formulas, device)
            n_correct_votes += correct_votes
            n_correct_assignments += correct_assignments
            n_satisfiable_problems += torch.sum(sats)

            sat_losses.append(sat_loss.item())
            assignment_losses.append(assignment_loss.item())

    voting_acc = n_correct_votes / len(dataloader.dataset)
    assignment_precision = n_correct_assignments / n_satisfiable_problems
    voting_loss = np.sum(sat_losses) / len(dataloader.dataset)
    assignment_loss = np.sum(assignment_losses) / len(dataloader.dataset)
 
    return voting_acc, assignment_precision, voting_loss, assignment_loss


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

    # init model
    model = NeuroSATAssign(config)
    assert config.training.optimizer.step_rule == "adam", "Adam is currently the only optimizer implemented"
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.optimizer.learning_rate, 
            weight_decay=config.training.optimizer.weight_decay)
    starting_epoch = 0

    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")


    if config.logging.load_checkpoint:
        # Save model, optimizer, current epoch
        checkpoint_path = "saved_models/{}.ckpt".format(config.logging.load_from_tag)
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if config.logging.resume_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1



    # Hyperparameters:
    sat_criterion = nn.BCELoss()
    assignment_criterion = NeuroSATLoss(config)
    
    history = []
    best_acc = -float('inf')

    for epoch in tqdm(range(starting_epoch, config.training.epochs), desc='Epoch Loop'):
        t_vote_acc, t_assignment_prec, t_vote_loss, t_assignment_loss = train_loop(
                config, train_dataloader, model, sat_criterion, assignment_criterion, optimizer, device, 
                "saved_models/{}_training.csv".format(config.logging.experiment_tag), epoch)
        
        v_vote_acc, v_assignment_prec, v_vote_loss, v_assignment_loss = val_loop(
                config, val_dataloader, model, sat_criterion, assignment_criterion, device)

        history.append({
            'epoch': epoch, 
            'train sat acc': t_vote_acc,
            'train assignment precision': t_assignment_prec,
            'train vote loss': t_vote_loss, 
            'train assignment loss': t_assignment_loss,
            'val sat acc': v_vote_acc,
            'val assignment precision': v_assignment_prec,
            'val vote loss': v_vote_loss, 
            'val assignment loss': v_assignment_loss,

        })
        if config.logging.save_checkpoint:
            checkpoint_path = "saved_models/{}.ckpt".format(config.logging.experiment_tag)
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
        if best_acc < v_vote_acc:
            best_acc = v_vote_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vote_acc': v_vote_acc,
                'assignment_prec': v_assignment_prec
            }, "saved_models/{}_best_model.ckpt".format(config.logging.experiment_tag))


        history_df = pd.DataFrame(history)
        append_df_to_csv(history_df, "saved_models/{}_epochs.csv".format(config.logging.experiment_tag))

