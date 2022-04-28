import argparse
import csv
from itertools import combinations
from functools import cmp_to_key
import yaml
import json
import os
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from dataset import MSLR10KDataset
from torch.utils.data import DataLoader
from torch import nn
from ranking_models import DirectRanker
from utilities import compute_acc, combinations_2, ndcg_score
from discriminators import discriminator_types
from relaxations import relaxation_types

def append_dict_to_csv(results_dict, csv_path, sep=","):
    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=list(results_dict.keys()))
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()    
        writer.writerow(results_dict)
    

def train_step(step_no: int, data_in: any, model: nn.Module, optimizer: any, criterion: any, config, device=torch.device("cpu")):

    optimizer.zero_grad()
    features, targets = data_in
    features, targets = features.float().to(device), targets.float().to(device)
    assert not torch.any(torch.isnan(features))

    if config.general.mode == "pair":
        with torch.cuda.amp.autocast():
            predictions = model(features).squeeze() # (B, 2, D) -> (B, 1) -> (B,)

            loss = criterion(predictions, targets)
            loss.backward()
            if not config.general.run_eval_mode:
                if config.training.optimizer.use_grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
            optimizer.step()    
            
        predictions = torch.where(predictions > 0, 1., 0.)
        correct = predictions == targets

        results = {
            'step': step_no,
            'accuracy': (torch.sum(correct) / len(correct)).item(), 
            'loss': loss.item(),
            'ndcg': 0,
        }
    elif config.general.mode == "list":
        B, N, D = features.shape
        pairwise_indices = combinations_2(np.arange(N), batched=False) # (NC2, 2)
        pairwise_features = combinations_2(features) # (B, NC2, 2, D)
        pairwise_targets = combinations_2(targets) # (B, NC2, 2)

        pairwise_targets = torch.where(pairwise_targets[:,:,0] > pairwise_targets[:,:,1], 1., 0.) # (B, NC2)
        pairwise_predictions = model(pairwise_features).squeeze() # (B, NC2, 2, D) -> (B, NC2, 1) -> (B, NC2)
        
        # backprop
        loss = criterion(pairwise_predictions, pairwise_targets)
        loss.backward()
        if not config.general.run_eval_mode:
            if config.training.optimizer.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
        optimizer.step()    
        # backprop

        batch_ndcg = 0

        for batch in range(B):
            permuted_relevances = targets[batch] # (N,)
            model_predictions = pairwise_predictions[batch] # (NC2)
            lookup_table = np.zeros((N, N))
            for idx, (x, y) in enumerate(pairwise_indices):
                lookup_table[x, y] = model_predictions[idx]
                lookup_table[y, x] = -1 * model_predictions[idx]

            def pairwise_comparator(x, y):
                return lookup_table[x, y]

            relevances_argsort = sorted(np.arange(N), key=cmp_to_key(pairwise_comparator), reverse=True)
            resorted_relevances = permuted_relevances[relevances_argsort]
            batch_ndcg += ndcg_score(resorted_relevances, device=device)
            

        results = {
            'step': step_no,
            'accuracy': 0, 
            'loss': loss.item(),
            'ndcg': (batch_ndcg / B).item(),
        }
    else:
        raise NotImplementedError()

    return step_no + 1, results



def run_epoch(step_no, dataloader, model, optimizer, criterion, config, device):

    accuracies = []
    losses = []
    ndcgs = []
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
                ndcgs.append(results['ndcg'])
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
        'epoch ndcg': np.mean(ndcgs),
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
    dataset = MSLR10KDataset(root_dir=config.general.dataset_root, folds=config.general.folds,
        partition=config.general.partition, seed=config.general.seed, mode=config.general.mode, k=config.general.k)

    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle,
            num_workers=config.training.num_workers)

    # init model
    model = DirectRanker(config)

    #assert config.training.optimizer.step_rule == "AdamW", "AdamW is currently the only optimizer supported"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.optimizer.learning_rate, 
            weight_decay=config.training.optimizer.weight_decay)

    # Squared sign loss, sign loss, maybe 0-1 loss?
    #criterion = lambda pred, target: torch.mean(torch.square(torch.minimum(pred * target, torch.zeros_like(pred))))
    #criterion = lambda pred, target: torch.mean(torch.minimum(pred * target, torch.zeros_like(pred)))
    criterion = nn.BCEWithLogitsLoss()


    step_no = 0
    
    batch_csv_path = os.path.join("logs", "{}_running_results.csv".format(config.logging.experiment_tag))
    epoch_csv_path = os.path.join("logs", "{}_epoch_results.csv".format(config.logging.experiment_tag))
    with open(batch_csv_path, 'w+') as f:
        pass
    with open(epoch_csv_path, 'w+') as f:
        pass

    checkpoint_path = os.path.join("checkpoints", "{}.ckpt".format(config.logging.experiment_tag))
    for epoch in tqdm(range(config.training.epochs), desc='Epoch Loop'):
        step_no, epoch_results = run_epoch(step_no, dataloader, model, optimizer, criterion, config, device)
        append_dict_to_csv(epoch_results, epoch_csv_path)
        if config.logging.save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': epoch_results,
            }, checkpoint_path)
        
