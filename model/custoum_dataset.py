#%%
import torch
from torch.utils.data import Dataset
from model.MolEncoder import MolEncoder

class SolPropDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        # MolEncoder?
        encs = [MolEncoder(mol) for mol in sample['mols']] # solute -> the last 

        def enc_to_dict(enc):
            return {
                'f_atoms': torch.tensor(enc.f_atoms, dtype=torch.float),
                'f_bonds': torch.tensor(enc.f_bonds, dtype=torch.float),
                'f_mols': torch.tensor(enc.f_mol, dtype=torch.float),
                'a2b': enc.a2b,
                'b2a': enc.b2a,
                'b2revb': enc.b2revb,
                'ascope': enc.ascope,
                'bscope': enc.bscope
            }

        dicts = [enc_to_dict(enc) for enc in encs]

        return {
            'solv1_name':sample['solv1_name'],
            'solv2_name':sample['solv2_name'],
            'solu_name':sample['solu_name'],
            'solute': dicts[-1],
            'solvent_list': dicts[:-1],
            'mol_frac': torch.tensor(sample['mol_fractions'], dtype=torch.float),
            'target': torch.tensor(sample['target'], dtype=torch.float)
        }

# %%