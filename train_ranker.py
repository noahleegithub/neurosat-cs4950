import argparse
from functools import cmp_to_key
import yaml
import json
import os
from types import SimpleNamespace
from sat_models import MaxSATLoss, NeuroMaxSAT
from tqdm import tqdm
import torch
import numpy as np

from dataset import MSLR10KDataset
from torch.utils.data import DataLoader
from torch import nn
from ranking_models import DirectRanker, MaxSATRanker
from utilities import average_precision, clause_accuracy, combinations_2, ndcg_score, append_dict_to_csv

def train_step(step_no: int, data_in: any, model: nn.Module, optimizer: any, criterion: any, config, device=torch.device("cpu")):

    optimizer.zero_grad()
    features, targets = data_in
    features, targets = features.float().to(device), targets.float().to(device)
    assert not torch.any(torch.isnan(features))

    if config.general.mode == "pair":
        with torch.cuda.amp.autocast():
            predictions = model(features).squeeze() # (B, 2, D) -> (B, 1) -> (B,)

            loss = criterion(predictions, targets)
            
            if not config.general.run_eval_mode:
                loss.backward()
                if config.training.optimizer.use_grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
                optimizer.step()    
            
        predictions = torch.where(predictions > 0, 1., 0.)
        correct = predictions == targets

        results = {
            'step': step_no,
            'accuracy': (torch.sum(correct) / len(correct)).item(), 
            'loss': loss.item(),
            'ndcg@{}'.format(config.general.k): 0,
            'map@{}'.format(config.general.k): 0,
        }
    elif config.general.mode == "list":
        B, N, D = features.shape

        with torch.cuda.amp.autocast():
            pairwise_indices = combinations_2(np.arange(N), batched=False) # (NC2, 2)
            pairwise_features = combinations_2(features) # (B, NC2, 2, D)
            pairwise_targets = combinations_2(targets) # (B, NC2, 2)

            # if x > y, 1. if x == y, 0.5, if x < y, 0.
            pairwise_targets_comp = pairwise_targets[:,:,0] - pairwise_targets[:,:,1]
            pairwise_targets_comp[pairwise_targets_comp > 0] = 1.0
            pairwise_targets_comp[pairwise_targets_comp == 0] = 0.5
            pairwise_targets_comp[pairwise_targets_comp < 0] = 0.0
            
            pair_comparisons, predicted_relevances_1, predicted_relevances_2 = model(pairwise_features) # (B, NC2, 2, D) -> (B, NC2, 1) -> (B, NC2)
            pair_comparisons, predicted_relevances_1, predicted_relevances_2 = pair_comparisons.squeeze(), predicted_relevances_1.squeeze(), predicted_relevances_2.squeeze()
            # backprop

            _, inverse_indices = np.unique(pairwise_indices, return_index=True)
            predicted_relevances = torch.cat((predicted_relevances_1, predicted_relevances_2), axis=1)[:, inverse_indices]
            targets_ndcg = targets.int()
            targets_map = targets.int()
            targets_ndcg = targets_ndcg.cpu().apply_(lambda x: vars(config.ndcg_hash)[str(x)])
            targets_map = targets_map.cpu().apply_(lambda x: vars(config.map_hash)[str(x)])
            targets_ndcg = targets_ndcg.float().to(device)
            targets_map = targets_map.float().to(device)

            loss = criterion(predicted_relevances, targets_ndcg.float().to(device))
            
            if not config.general.run_eval_mode:
                loss.backward()
                if config.training.optimizer.use_grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
                optimizer.step()    
        # backprop

        batch_ndcgs = []
        batch_aps = []

        for batch in range(B):
            model_predictions = pair_comparisons[batch] # (NC2)
            lookup_table = np.zeros((N, N))
            for idx, (x, y) in enumerate(pairwise_indices):
                lookup_table[x, y] = model_predictions[idx]
                lookup_table[y, x] = -1 * model_predictions[idx]

            def pairwise_comparator(x, y):
                return lookup_table[x, y]

            relevances_argsort = sorted(np.arange(N), key=cmp_to_key(pairwise_comparator), reverse=True)
            resorted_relevances = predicted_relevances[batch][relevances_argsort]
            batch_ndcgs.append(ndcg_score(resorted_relevances.detach().float(), optimal_ranking=targets_ndcg[batch][relevances_argsort], k=config.general.k))
            batch_aps.append(average_precision(targets_map[batch][relevances_argsort], k=config.general.k))

        if config.general.debug and step_no % config.general.debug_freq == 0: 
            print(resorted_relevances, batch_ndcgs[-1], batch_aps[-1])

        predictions = pair_comparisons[:]
        predictions[pair_comparisons > 0] = 1.0
        predictions[pair_comparisons == 0] = 0.5
        predictions[pair_comparisons < 0] = 0.0
        correct = predictions == pairwise_targets_comp  

        results = {
            'step': step_no,
            'accuracy': (torch.sum(correct) / torch.numel(correct)).item(), 
            'loss': loss.item(),
            'ndcg@{}'.format(config.general.k): np.nanmean(batch_ndcgs),
            'map@{}'.format(config.general.k): np.nanmean(batch_aps),
        }
    elif config.general.mode == "maxsat":
        B, N, D = features.shape
        pairwise_criterion, maxsat_criterion = criterion
        pairwise_indices = combinations_2(np.arange(N), batched=False) # (NC2, 2)


        with torch.cuda.amp.autocast():
            pairwise_targets = combinations_2(targets) # (B, NC2, 2)

            # if x > y, 1. if x == y, 0.5, if x < y, 0.
            pairwise_targets_comp = pairwise_targets[:,:,0] - pairwise_targets[:,:,1]
            pairwise_targets_comp[pairwise_targets_comp > 0] = 1.0
            pairwise_targets_comp[pairwise_targets_comp == 0] = 0.5
            pairwise_targets_comp[pairwise_targets_comp < 0] = 0.0
        
            ranker_predictions, maxsat_assignments, formulas = model(features)

            pairwise_loss = pairwise_criterion(ranker_predictions, pairwise_targets_comp)
            sat_loss = maxsat_criterion(maxsat_assignments, formulas)
            
            if not config.general.run_eval_mode:
                (pairwise_loss * config.training.pairwise_alpha).backward(retain_graph=True)
                (sat_loss * config.training.sat_alpha).backward()
                if config.training.optimizer.use_grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.optimizer.clip_grad_norm)
                optimizer.step()   
            
            loss = config.training.pairwise_alpha * pairwise_loss + config.training.sat_alpha * sat_loss

        batch_ndcgs = []
        batch_aps = []

        for batch in range(B):
            permuted_relevances = targets[batch] # (N,)
            model_predictions = maxsat_assignments[batch] # (NC2)
            lookup_table = np.zeros((N, N))
            for idx, (x, y) in enumerate(pairwise_indices):
                lookup_table[x, y] = model_predictions[idx]
                lookup_table[y, x] = -1 * model_predictions[idx]

            def pairwise_comparator(x, y):
                return lookup_table[x, y]

            relevances_argsort = sorted(np.arange(N), key=cmp_to_key(pairwise_comparator), reverse=True)
            resorted_relevances = permuted_relevances[relevances_argsort]
            batch_ndcgs.append(ndcg_score(resorted_relevances, vars(config.ndcg_hash), k=config.general.k))
            batch_aps.append(average_precision(resorted_relevances, vars(config.map_hash), k=config.general.k))

        if config.general.debug and step_no % config.general.debug_freq == 0: 
            print(maxsat_assignments[-1])
            print(resorted_relevances, batch_ndcgs[-1], batch_aps[-1])            

        batch_acc = clause_accuracy(maxsat_assignments, formulas, device)

        results = {
            'step': step_no,
            'accuracy': batch_acc.item(), 
            'loss': loss.item(),
            'ndcg@{}'.format(config.general.k): np.nanmean(batch_ndcgs),
            'map@{}'.format(config.general.k): np.nanmean(batch_aps),
        }

    return step_no + 1, results



def run_epoch(step_no, dataloader, model, optimizer, criterion, config, device):

    accuracies = []
    losses = []
    ndcgs = []
    maps = []
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
                ndcgs.append(results['ndcg@{}'.format(config.general.k)])
                maps.append(results['map@{}'.format(config.general.k)])
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
        'epoch ndcg@{}'.format(config.general.k): np.mean(ndcgs),
        'epoch map@{}'.format(config.general.k): np.mean(maps),
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
        partition=config.general.partition, seed=config.general.seed, mode=config.general.mode, 
        k=config.general.list_length)

    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle,
            num_workers=config.training.num_workers)

    # init model
    model = DirectRanker(config)

    if config.general.mode == "maxsat":
        constraint_solver = NeuroMaxSAT(config)
        model = MaxSATRanker(config, model, constraint_solver)

    #assert config.training.optimizer.step_rule == "AdamW", "AdamW is currently the only optimizer supported"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.optimizer.learning_rate, 
            weight_decay=config.training.optimizer.weight_decay)

    # Squared sign loss, sign loss, maybe 0-1 loss?
    #criterion = lambda pred, target: torch.mean(torch.square(torch.minimum(pred * target, torch.zeros_like(pred))))
    #criterion = lambda pred, target: torch.mean(torch.minimum(pred * target, torch.zeros_like(pred)))
    criterion = nn.MSELoss()
    
    if config.general.mode == "maxsat":
        criterion = (criterion, MaxSATLoss(config))
        pass

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
        
