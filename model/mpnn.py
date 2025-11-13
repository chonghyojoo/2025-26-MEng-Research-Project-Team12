import torch
import torch.nn as nn
import torch.nn.functional as F
from solvation_predictor.models.MPN import MPN
from types import SimpleNamespace

def to_namespace(obj):
        # dict to SimpleNamespace
        return obj if isinstance(obj, SimpleNamespace) else SimpleNamespace(**obj)

class SolvationModel(nn.Module):
    def __init__(self, args):
        super(SolvationModel, self).__init__()

        self.args = args
        self.atom_output_dim = args.hidden_size

        # D-MPNN encoder for solute and solvent
        self.solute_encoder = MPN(args)
        self.solvent_encoder = MPN(args)

        # Fully connected layers for solute and solvent
        self.fc_solute = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fc_solvent = nn.Linear(args.hidden_size, args.ffn_hidden_size)

        # Final feedforward layers
        self.ffn = nn.Sequential(
            nn.Linear(args.ffn_hidden_size*2, args.ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(args.ffn_hidden_size, args.output_size)
        )
    def forward(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_solute_vecs = []
        batch_solvent_vecs = []

        # Fixed Temp.
        for solute_dict, solvent_dicts, mol_fracs in zip(batch['solute'], batch['solvent_list'], batch['mol_frac']):
            solute_ns = to_namespace(solute_dict)
            solute_vec, _ = self.solute_encoder(solute_ns)
            solute_vec = self.fc_solute(solute_vec.to(device))

            solvent_vecs = []
            for solvent_dict in solvent_dicts:
                solvent_ns = to_namespace(solvent_dict)
                solvent_vec, _ = self.solvent_encoder(solvent_ns)
                solvent_vec = self.fc_solvent(solvent_vec)
                # solvent_vec = solvent_vec.squeeze(0) # dim match
                solvent_vecs.append(solvent_vec)
                
            solvent_vecs = torch.stack(solvent_vecs, dim=0).to(device)
            # solute_vecs = torch.stack(solute_vec, dim=0).to(device)
            mol_fracs = mol_fracs.to(device)# unsqueeze(0)
            
            weighted_solvent = torch.matmul(mol_fracs, solvent_vecs.squeeze(1))
            batch_solvent_vecs.append(weighted_solvent.squeeze(0))
            batch_solute_vecs.append(solute_vec.squeeze(0))
            
        solute_batch = torch.stack(batch_solute_vecs, dim=0)
        solvent_batch = torch.stack(batch_solvent_vecs, dim=0)
        x = torch.cat([solute_batch, solvent_batch], dim=1) 

        out = self.ffn(x)

        return out
