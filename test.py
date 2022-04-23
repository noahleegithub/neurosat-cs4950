import argparse
import os
from tqdm import tqdm
import torch
from dataset import NeuroSATDataset
from torch.utils.data import DataLoader
from dataset import collate_adjacencies 
from models import NeuroSATAssign, NeuroSATLoss
from utilities import compute_acc

def test_loop(dataloader, model, device, log_file=None):
    if log_file is not None:
        fd = os.open(log_file, os.O_RDWR|os.O_CREAT)
    else:
        fd = 1
    model.eval()
    model.to(device)
    size = len(dataloader.dataset)
    total_acc = 0
    for batch_n, X in tqdm(enumerate(dataloader), total=len(dataloader)):
        adj_mtrx, lit_counts, discriminators = X
        adj_mtrx = adj_mtrx.to(device)
  
        preds, lit_counts = model(adj_mtrx, lit_counts, device=device)
        batch_acc = compute_acc(preds, lit_counts, discriminators)
        total_acc += batch_acc
   

        if batch_n % 100 == 0:
            current = batch_n * (size // len(dataloader))
            os.write(fd, f"batch acc: {batch_acc:>7f}  [{current}/{size}]\n".encode('utf-8'))
    
    total_acc = total_acc / len(dataloader)
    os.write(fd, f"overall acc: {total_acc:>7f}\n".encode('utf-8'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help="Path of the dataset to be used for training.")
    parser.add_argument('trained_model', help="Path to the pretrained model.")
    parser.add_argument('--batch-size', dest='batch_size', help="Batch size for training.", type=int, default=1)
    parser.add_argument('--shuffle', dest='shuffle', help="Reshuffle the dataset every epoch.", action='store_true')
    parser.add_argument('--num-workers', dest='num_workers', help="Number of dataloader workers.", type=int, default=0)
    parser.add_argument('--embed-dim', dest='embedding_dim', help="Dimension of model embeddings.", type=int, default=128)
    parser.add_argument('--lstm-itrs', dest='lstm_iterations', help="The number of message passing iterations.",
            type=int, default=26)
    parser.add_argument('--mlp-n-layers', dest='mlp_n_layers', help="Number of layers in each multilayer perceptron.",
            type=int, default=3)
    parser.add_argument('--log-file', dest='log_file', help="File for logging loss during testing.", default=None)
    parser.add_argument('--gpu', action='store_true', help="Run the training with GPU acceleration.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
   
    # create dataset
    test_dataset = NeuroSATDataset(root_dir=args.dataset_path, train=False, discriminator="minisat")
    # create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
            num_workers=args.num_workers, collate_fn=collate_adjacencies)
    # init model
    neuro_sat = NeuroSATAssign(args.embedding_dim, args.lstm_iterations, args.mlp_n_layers)
    # Hyperparamers:
    #   epochs, batch size, learning rate
    neuro_sat.load_state_dict(torch.load(args.trained_model))
    
    # define optimizer
    # TODO: incorporate epochs
    test_loop(test_dataloader, neuro_sat, device, log_file=args.log_file)
    # for batch in dataloader:
    #   predict
    #   acc

