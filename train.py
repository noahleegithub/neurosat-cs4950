import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
from dataset import NeuroSATDataset, collate_adjacencies
from torch.utils.data import DataLoader
from models import NeuroSATAssign, NeuroSATLoss
from utilities import compute_acc
from discriminators import discriminator_types
from relaxations import relaxation_types

def train_loop(dataloader, model, loss_fn, optimizer, device, history, epoch):
    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0
    for _, data in enumerate(tqdm(dataloader, desc='Batch loop')):
        optimizer.zero_grad()
        adj_mtrx, lit_counts, discriminators = data
        adj_mtrx = adj_mtrx.to(device)
  
        preds, lit_counts = model(adj_mtrx, lit_counts, device=device)
        loss = loss_fn(preds, lit_counts, discriminators)

        loss.backward(retain_graph=True)
        optimizer.step()
        
        accuracy = compute_acc(preds, lit_counts, discriminators)
        train_acc += accuracy
        train_loss += loss.item()
        history.append({'epoch' : epoch, 'batch loss' : loss.item(), 'batch acc' : accuracy})

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    model.to(device)

    val_loss = 0
    val_acc = 0
    for _, data in enumerate(dataloader):
        optimizer.zero_grad()
        adj_mtrx, lit_counts, discriminators = data
        adj_mtrx = adj_mtrx.to(device)
  
        preds, lit_counts = model(adj_mtrx, lit_counts, device=device)
        loss = loss_fn(preds, lit_counts, discriminators)
        
        accuracy = compute_acc(preds, lit_counts, discriminators)
        val_acc += accuracy
        val_loss += loss.item()

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help="Path of the dataset to be used for training.")
    parser.add_argument('epochs', help="Number of iterations to train over the dataset", type=int)
    parser.add_argument('discriminator', help="The module used to verify satisfying assignments.", 
            choices=discriminator_types.keys())
    parser.add_argument('relaxation', help="The type of fuzzy relaxation to use in the discriminator", 
            choices=relaxation_types.keys())
    parser.add_argument('--batch-size', help="Batch size for training.", type=int, default=1)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=2e-5)
    parser.add_argument('--shuffle', help="Reshuffle the dataset every epoch.", action='store_true')
    parser.add_argument('--num-workers', help="Number of dataloader workers.", type=int, default=0)
    parser.add_argument('--embed-dim', dest='embedding_dim', help="Dimension of model embeddings.", type=int, default=128)
    parser.add_argument('--lstm-itrs', dest='lstm_iterations', help="The number of message passing iterations.", 
            type=int, default=26)
    parser.add_argument('--mlp-n-layers', help="Number of layers in each multilayer perceptron.", type=int, default=3)
    parser.add_argument('--log-csv', help="CSV file for logging loss and accuracy during training.", default=None)
    parser.add_argument('--gpu', action='store_true', help="Run the training with GPU acceleration.")
    parser.add_argument('--checkpoint', help="Restart training from a checkpointed model.", default=None)

    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
   
    # create dataset
    train_dataset = NeuroSATDataset(root_dir=args.dataset_path, partition="train", discriminator=args.discriminator, 
            relaxation=args.relaxation)
    val_dataset = NeuroSATDataset(root_dir=args.dataset_path, partition="validation", discriminator="minisat",
            relaxation=args.relaxation)
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
            num_workers=args.num_workers, collate_fn=collate_adjacencies)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
            num_workers=args.num_workers, collate_fn=collate_adjacencies)
    # init model
    neuro_sat = NeuroSATAssign(args.embedding_dim, args.lstm_iterations, args.mlp_n_layers)
    if args.checkpoint is not None:
        neuro_sat.load_state_dict(torch.load(args.checkpoint))
    # Hyperparamers:
    #   epochs, batch size, learning rate

    # optimization loop
    optimizer = torch.optim.Adam(neuro_sat.parameters(), lr=args.learning_rate)
    # define loss function
    loss_fn = NeuroSATLoss()
    
    history = []    
    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    model_save_str = "./saved_models/NeuroSAT_epochs={}_dataset={}_batch-size={}_lr={}_discrim={}_relaxation={}_embed-dim={}_lstm-itrs={}_mlp-layers={}.pk".format(
            args.epochs, dataset_name, args.batch_size, 
            args.learning_rate, args.discriminator, args.relaxation,
            args.embedding_dim, args.lstm_iterations, args.mlp_n_layers)
    # define optimizer
    # TODO: record batch level loss, add restarting from checkpoint
    best_acc = -float('inf')
    for epoch in tqdm(range(args.epochs), desc='Epoch loop'):
        train_loss, train_acc = train_loop(train_dataloader, neuro_sat, loss_fn, optimizer, device, history, epoch)
        val_loss, val_acc = val_loop(val_dataloader, neuro_sat, loss_fn, device)
        history.append({"epoch" : epoch, "train loss" : train_loss, "train acc" : train_acc,
            "val loss" : val_loss, "val acc" : val_acc})
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(neuro_sat.state_dict(), model_save_str)


    history = pd.DataFrame(history)

    if args.log_csv is not None:
        history.to_csv(os.path.join("logs", args.log_csv))


