from typing import Dict, Any, List, Optional, Tuple
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import os
from model.mpnn import SolvationModel
from model.custoum_dataset import SolPropDataset

class Args:
    def __init__(self):
        self.hidden_size = 200
        self.ffn_hidden_size = 100
        self.output_size = 2
        self.dropout = 0.1
        self.bias = True
        self.depth = 2
        self.activation = "ReLU"
        self.cuda = True
        self.property = "solvation"
        self.aggregation = "mean"
        self.atomMessage = True

def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: 1D numpy arrays
    returns: dict with RMSE, MAE, R2, MAPE
    """
    import numpy as np
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    diff = y_pred - y_true
    mse = np.mean(diff**2)
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))

    # R^2 (manual to avoid sklearn dependency)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    # MAPE (safe)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE (%)": mape}


# ==============================
# Helper
# ==============================
def to_namespace(obj):
    return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

@torch.no_grad()
def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """dataset to collate batch model"""
    batch = dict(batch)  # shallow copy
    batch['mol_frac'] = batch['mol_frac'].to(device)
    batch['target']   = batch['target'].to(device)

    # solute dicts
    solutes = []
    for solute in batch['solute']:
        ns = to_namespace(solute)
        for k, v in vars(ns).items():
            if isinstance(v, torch.Tensor):
                setattr(ns, k, v.to(device))
        solutes.append(ns)
    batch['solute'] = solutes

    # solvent dicts
    out = []
    for solvent_list in batch['solvent_list']:
        tmp = []
        for solvent in solvent_list:
            ns = to_namespace(solvent)
            for k, v in vars(ns).items():
                if isinstance(v, torch.Tensor):
                    setattr(ns, k, v.to(device))
            tmp.append(ns)
        out.append(tmp)
    batch['solvent_list'] = out
    return batch

def collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """without DataLoader, 1~N samples can be used"""
    import torch
    return {
        'solute':       [it['solute'] for it in items],                     # list of dict
        'solvent_list': [it['solvent_list'] for it in items],               # list of list of dict
        'mol_frac':     torch.stack([it['mol_frac'] for it in items]),      # (B, 2)
        'target':       torch.stack([it['target'] for it in items]),        # (B,2) 
        'name' :        [ (it.get('solv1_name','unknown'),
                           it.get('solv2_name','unknown'),
                           it.get('solu_name','unknown')) for it in items ]
    }
# Name extraction
def extract_names_from_sample(sample):
    """
    sample: dataset[i] element(dict)
    return: (s1, s2, su) — solvent1/2, solute SMILES (if no name, 'unknown')
    """
    s1 = sample.get('solv1_name', 'unknown')
    s2 = sample.get('solv2_name', 'unknown')
    su = sample.get('solu_name',  'unknown')
    return s1, s2, su

def extract_names_from_batch(batch, index=0):
    if 'names' in batch:
        s1, s2, su = batch['names'][index]
        return s1, s2, su
    return ('unknown', 'unknown', 'unknown')

def load_model(ckpt_path: str,
               device: Optional[torch.device] = None,
               args: Optional[Args] = None) -> SolvationModel:
    """Model load"""
    if args is None:
        args = Args()
    if device is None:
        device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    model = SolvationModel(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_dataset(pkl_path: str) -> SolPropDataset:
    """preprocessed 'pkl' load -> Dataset """
    import pickle, os
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found.")
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)
    return SolPropDataset(data_list)

@torch.no_grad()
def predict_single(model: SolvationModel,
                   sample: Dict[str, Any],
                   device: Optional[torch.device] = None) -> float:
    """single sample of dataset[i], pred G_solv"""
    if device is None:
        device = next(model.parameters()).device
    batch = collate_fn([sample])
    batch = batch_to_device(batch, device)
    y_hat = model(batch)  # (1,1)
    return y_hat.detach().cpu().numpy()

@torch.no_grad()
def predict_batch(model: SolvationModel,
                  samples: List[Dict[str, Any]],
                  device: Optional[torch.device] = None) -> np.ndarray:
    """batch inference to (B,) numpy"""
    if device is None:
        device = next(model.parameters()).device
    batch = collate_fn(samples)
    batch = batch_to_device(batch, device)
    y_hat = model(batch)  # (B,1)
    return y_hat.detach().cpu().numpy()

