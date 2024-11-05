import os
import yaml
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shutil import copy
from tqdm import tqdm
from utils import set_seed, tsne_visualization
from model import Model
from metrics import compute_metrics
from data_utils import LandmarkVideoDataset


def main(args):
    """
    Main function to run the pipeline
    """
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed for reproducibility
    set_seed(config)

    model_config = config["model_config"]

    # Create experiment directory
    exp_tag = f"{args.comment}_{config['epochs']}_{config['batch_size']}_{config['lr']}"
    os.makedirs("exp_results", exist_ok=True)
    exp_folder= f"exp_results/{exp_tag}"
    os.makedirs(exp_folder, exist_ok=True)

    # Set files in experiment directory
    copy(args.config, f"{exp_folder}/config.yaml")
    ckpt_folder = f"{exp_folder}/checkpoints"
    os.makedirs(ckpt_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_config).to(device)
    nb_params = sum([param.view(-1).size(0) for param in model.parameters()])
    print(f"Number of parameters: {nb_params}")

    wandb.init(project=f'FG_{exp_tag}', name=exp_tag, config=config)
    wandb.watch(model)
    wandb.define_metric('train_loss', summary='min')
    wandb.define_metric('train_auc', summary='max')
    wandb.define_metric('train_acc', summary='max')
    wandb.define_metric('val_loss', summary='min')
    wandb.define_metric('val_auc', summary='max')
    wandb.define_metric('val_acc', summary='max')
    wandb.define_metric('test_loss', summary='min')
    wandb.define_metric('test_auc', summary='max')
    wandb.define_metric('test_acc', summary='max')
    
    # Load data
    trainloader = get_loader(config, type='train')
    valloader = get_loader(config, type='train')
    testloader = get_loader(config, type='train')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Train Model
    for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress"):
        train_model(model, trainloader, criterion, optimizer, epoch, ckpt_folder, device)
        evaluate_model(model, valloader, criterion, epoch, exp_folder, 'val', config['seed'], device)
        evaluate_model(model, testloader, criterion, epoch, exp_folder, 'test', config['seed'], device)

    wandb.finish()

def get_loader(config, type):
    """
    Returns a DataLoader based on the dataset type ('train', 'val', 'eval').
    Removes DDP-related elements and uses a simple DataLoader.
    """
    csv_file_path = config["csv"]
    num_workers = config["num_workers"]
    if type == 'train':
        dataset = LandmarkVideoDataset(data=csv_file_path, dataset_type=type)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers
        )
    elif type == 'val':
        dataset = LandmarkVideoDataset(data=csv_file_path, dataset_type=type)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers
        )
    elif type == 'eval':
        dataset = LandmarkVideoDataset(data=csv_file_path, dataset_type=type)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers
        )
    
    return loader

def train_model(model, dataloader, criterion, optimizer, epoch, ckpt_folder, device):
    model.train()
    
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # pv: patch video, l: landmark, gv: global video, y: label
    for pv, l, gv, _, y in tqdm(dataloader, desc="Training for epoch {}".format(epoch+1), total=len(dataloader)):
        pv, l, gv, y = pv.to(device), l.to(device), gv.to(device), y.view(-1, 1).type(torch.float32).to(device)

        output, _ = model(pv, l, gv)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        # Store labels and predictions for metrics
        all_labels.extend(y.cpu().numpy())  # Convert labels to numpy and store
        all_preds.extend(torch.sigmoid(output).detach().cpu().numpy())  # Get probability of class 1

    # Compute average loss
    avg_loss = running_loss / len(dataloader)
    auc, acc = compute_metrics(all_labels, all_preds)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss}, AUC: {auc}, Accuracy: {acc}")
    step = epoch*3
    wandb.log({"train_loss": avg_loss, "train_auc": auc, "train_acc": acc}, step=step)

    # Save model
    torch.save(model.state_dict(), f"{ckpt_folder}/ckpt_{epoch+1}_{int(acc*100)}.pt")

def evaluate_model(model, dataloader, criterion, epoch, exp_folder, type, seed, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_ids = []
    all_labels = []
    all_preds = []
    all_hidden = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for pv, l, gv, id, y in tqdm(dataloader, desc=f"{type} for Epoch {epoch+1}", total=len(dataloader)):
            # Move inputs and labels to the specified device
            pv, l, gv, y = pv.to(device), l.to(device), gv.to(device), y.view(-1, 1).type(torch.float32).to(device)
            
            # Forward pass
            output, last_hidden = model(pv, l, gv)
            loss = criterion(output, y)
            
            # Accumulate loss
            running_loss += loss.item()

            # Store id, label, prediction
            all_ids.extend(id)
            all_labels.extend(y.cpu().numpy())  # Convert labels to numpy and store
            all_preds.extend(torch.sigmoid(output).detach().cpu().numpy())  # Get probability of class 1
            all_hidden.extend(last_hidden.cpu().numpy())

    # Compute metrics
    avg_loss = running_loss / len(dataloader)
    auc, acc = compute_metrics(all_labels, all_preds)
    # Log aggregated results per epoch in W&B
    if type == 'test':
        step=epoch*3+2
        print(f"Epoch {epoch+1} Test Loss: {avg_loss}, AUC: {auc}, Accuracy: {acc}")
        wandb.log({
            "test_loss": avg_loss,
            "test_auc": auc,
            "test_acc": acc,
            "epoch": epoch,
            f"{type}_ids": all_ids,
            f"{type}_preds": all_preds,
            f"{type}_labels": all_labels
        }, step=step)
    else:
        step=epoch*3+1
        print(f"Epoch {epoch+1} Validation Loss: {avg_loss}, AUC: {auc}, Accuracy: {acc}")
        wandb.log({
            "val_loss": avg_loss,
            "val_auc": auc,
            "val_acc": acc,
            "epoch": epoch,
            f"{type}_ids": all_ids,
            f"{type}_preds": all_preds,
            f"{type}_labels": all_labels
        }, step=step)

    # Visualize with TSNE after evaluation
    os.makedirs(f"{exp_folder}/tsne", exist_ok=True)
    tsne_visualization(np.array(all_hidden), np.array(all_labels), epoch, exp_folder, type=type, plot_title=f"{type} Set t-SNE Visualization", seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='path to config file')
    parser.add_argument('--comment', type=str, default='', help='comment for the experiment')
    args = parser.parse_args()
    main(args)