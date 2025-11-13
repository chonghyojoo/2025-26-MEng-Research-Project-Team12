# %%
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
#%%
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from model.mpnn import SolvationModel
import numpy as np
from model.custoum_dataset import SolPropDataset
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
import pickle
from tqdm import tqdm
import random
#%%
# reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

#%%
class Args:
    def __init__(self):
        self.hidden_size = 200             # D-MPNN hidden dim
        self.ffn_hidden_size = 100         # Feedforward hidden dim
        self.output_size = 2               # # of properties
        self.dropout = 0.1                 # 
        self.bias = True
        self.depth = 2                     
        self.activation = "ReLU"           # activation
        self.cuda = True                   # GPU
        self.property = "solvation"
        self.aggregation = "mean"
        self.atomMessage = False           # False: only atom

def collate_fn(batch):
    batched_data = {
        'solute': [item['solute'] for item in batch],                 # list of dict
        'solvent_list': [item['solvent_list'] for item in batch],     # list of list of dict
        'mol_frac': torch.stack([item['mol_frac'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])     # (B, 2)
    }
    return batched_data

def to_namespace(obj):
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)
#%%
import os 
os.getcwd()

#%%
# load processed dataset (binary by default)
data_path = "./preprocessing/processed_binarysolv_exp.pkl"   
with open(data_path, 'rb') as f:
    data_list = pickle.load(f)

#%%
# 2) create dataset and dataloaders
dataset = SolPropDataset(data_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#%%
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

#%%
# model and optimizer
args = Args()
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
model = SolvationModel(args).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='none')

#%%
# Tensorboard 
writer = SummaryWriter(log_dir='runs/solprop_test')

# move dict tensors to device
def move_batch_to_device(batch, device):
    batch['mol_frac'] = batch['mol_frac'].to(device)
    batch['target']   = batch['target'].to(device)

    # Move solutes
    solutes = []
    for solute in batch['solute']:
        solute_ns = to_namespace(solute)
        for k, v in vars(solute_ns).items():
            if isinstance(v, torch.Tensor):
                setattr(solute_ns, k, v.to(device))
        solutes.append(solute_ns)
    batch['solute'] = solutes

    # Move solvents
    solvents_out = []
    for solvent_list in batch['solvent_list']:
        tmp = []
        for solvent in solvent_list:
            solvent_ns = to_namespace(solvent)
            for k, v in vars(solvent_ns).items():
                if isinstance(v, torch.Tensor):
                    setattr(solvent_ns, k, v.to(device))
            tmp.append(solvent_ns)
        solvents_out.append(tmp)
    batch['solvent_list'] = solvents_out
    return batch

# metrics
def rmse(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()
def mae(y_hat, y):
    return torch.mean(torch.abs(y_hat - y)).item()

#%%
# train loop 
best_val_loss = float('inf')
patience = 10
epochs_without_improve = 0
max_epochs = 100

for epoch in range(max_epochs):
    # === Training ===
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    train_mae  = 0.0
    n_train    = 0

    train_gsolv_loss = 0.0
    train_gsolv_rmse = 0.0
    train_gsolv_mae = 0.0

    train_hsolv_loss = 0.0
    train_hsolv_rmse = 0.0
    train_hsolv_mae = 0.0
    

    tepoch = tqdm(train_loader, unit='batch', desc=f"[Train] Epoch {epoch+1}/{max_epochs}")
    for batch in tepoch:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(batch)   # (B, 2)
        target = batch['target']  # (B, 2)                     

        loss_per_target = criterion(output, target)  # (B, 2)
        loss_Gsolv = loss_per_target[:, 0]
        loss_Hsolv = loss_per_target[:, 1]
        
        loss = loss_Gsolv.mean() + loss_Hsolv.mean()
        loss.backward()
        optimizer.step()

        B = output.size(0)
        n_train += B
        train_loss += loss.item() * B
        train_rmse += rmse(output.detach(), target)
        train_mae  += mae(output.detach(), target)

        train_gsolv_loss += loss_Gsolv.mean().item() * B
        train_gsolv_rmse += rmse(output[:, 0].detach(), target[:, 0])
        train_gsolv_mae  += mae(output[:, 0].detach(), target[:, 0])

        train_hsolv_loss += loss_Hsolv.mean().item() * B
        train_hsolv_rmse += rmse(output[:, 1].detach(), target[:, 1])
        train_hsolv_mae  += mae(output[:, 1].detach(), target[:, 1])

        tepoch.set_postfix(loss=loss.item(), refresh=False)

    train_loss /= n_train
    train_rmse /= len(train_loader)
    train_mae  /= len(train_loader)

    train_gsolv_loss /= n_train
    train_gsolv_rmse /= len(train_loader)
    train_gsolv_mae /= len(train_loader)

    train_hsolv_loss /= n_train
    train_hsolv_rmse /= len(train_loader)
    train_hsolv_mae /= len(train_loader)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("RMSE/train", train_rmse, epoch)
    writer.add_scalar("MAE/train",  train_mae,  epoch)

    writer.add_scalar("Loss_Gsolv/train", train_gsolv_loss, epoch)
    writer.add_scalar("RMSE_Gsolv/train", train_gsolv_rmse, epoch)
    writer.add_scalar("MAE_Gsolv/train", train_gsolv_mae, epoch)

    writer.add_scalar("Loss_Hsolv/train", train_hsolv_loss, epoch)
    writer.add_scalar("Loss_Hsolv/train", train_hsolv_rmse, epoch)
    writer.add_scalar("Loss_Hsolv/train", train_hsolv_mae, epoch)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    val_mae  = 0.0
    n_val    = 0

    val_gsolv_loss = 0.0
    val_gsolv_rmse = 0.0
    val_gsolv_mae = 0.0

    val_hsolv_loss = 0.0
    val_hsolv_rmse = 0.0
    val_hsolv_mae = 0.0

    vepoch = tqdm(val_loader, unit='batch', desc=f"[Val]   Epoch {epoch+1}/{max_epochs}")
    with torch.no_grad():
        for batch in vepoch:
            batch = move_batch_to_device(batch, device)

            output = model(batch)
            target = batch['target']
            loss_per_target = criterion(output, target)

            loss_Gsolv = loss_per_target[:, 0]
            loss_Hsolv = loss_per_target[:, 1]
            loss = loss_Gsolv.mean() + loss_Hsolv.mean()

            B = output.size(0)
            n_val += B
            val_loss += loss.item() * B
            val_rmse += rmse(output.detach(), target)
            val_mae  += mae(output.detach(), target)

            val_gsolv_loss += loss_Gsolv.mean().item() * B
            val_gsolv_rmse += rmse(output[:, 0].detach(), target[:, 0])
            val_gsolv_mae  += mae(output[:, 0].detach(), target[:, 0])

            val_hsolv_loss += loss_Hsolv.mean().item() * B
            val_hsolv_rmse += rmse(output[:, 1].detach(), target[:, 1])
            val_hsolv_mae  += mae(output[:, 1].detach(), target[:, 1])

            vepoch.set_postfix(loss=loss.item(), refresh=False)

    val_loss /= n_val
    val_rmse /= len(val_loader)
    val_mae  /= len(val_loader)

    val_gsolv_loss /= n_val
    val_gsolv_rmse /= len(val_loader)
    val_gsolv_mae /= len(val_loader)

    val_hsolv_loss /= n_val
    val_hsolv_rmse /= len(val_loader)
    val_hsolv_mae /= len(val_loader)


    writer.add_scalar("Loss/val",  val_loss, epoch)
    writer.add_scalar("RMSE/val",  val_rmse, epoch)
    writer.add_scalar("MAE/val",   val_mae,  epoch)

    writer.add_scalar("Loss_Gsolv/val", val_gsolv_loss, epoch)
    writer.add_scalar("RMSE_Gsolv/val", val_gsolv_rmse, epoch)
    writer.add_scalar("MAE_Gsolv/val", val_gsolv_mae, epoch)

    writer.add_scalar("Loss_Hsolv/val", val_hsolv_loss, epoch)
    writer.add_scalar("Loss_Hsolv/val", val_hsolv_rmse, epoch)
    writer.add_scalar("Loss_Hsolv/val", val_hsolv_mae, epoch)

    print(f"Epoch {epoch:03d}: "
          f"Train Loss={train_loss:.4f} RMSE={train_rmse:.4f} MAE={train_mae:.4f} | "
          f"Val Loss={val_loss:.4f} RMSE={val_rmse:.4f} MAE={val_mae:.4f}")

    # Early Stopping
    if val_loss < best_val_loss - 1e-6:     
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

writer.close()

# %%
