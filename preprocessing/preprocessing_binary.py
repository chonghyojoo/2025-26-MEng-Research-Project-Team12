#%%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import pickle
#%%
mydir = os.getcwd()
print(mydir)

# %% data preprocessing --> from 'data' folder

data_path = './data/BinarySolvGH-QM.csv' 

# further: single solvent tratin - 'BinarySolvGH-QM.csv'

# test1: 'Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv'
# test2: 'Data and predictions of solvation free energies in ternary solvents (TernarySolv-Exp).csv'
df = pd.read_csv(data_path, encoding='cp1252')

# %%  
# function: InChI to SMILES
def inchi_to_smiles(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            mol = Chem.AddHs(mol)  # add H
            return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=False) # Chem.MolToSmiles(mol)
        else:
            return 'Invalid InChI'
    except Exception as e:
        return f'Error: {str(e)}'
    
# function: SMILES to Mol_obje
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        if mol.GetNumBonds() == 0:
            mol = Chem.AddHs(mol) 
        #AllChem.EmbedMolecule(mol)
        AllChem.Compute2DCoords(mol)
    return mol

# function: mol fraction list
def get_mol_list(row):
    mols = []
    fracs = []

    if pd.notnull(row['inchi_solvent1_smiles']):
        mols.append(row['solvent1_mol'])
        fracs.append(row['frac_solvent1'])
    if pd.notnull(row['inchi_solvent2_smiles']):
        mols.append(row['solvent2_mol'])
        fracs.append(1 - row['frac_solvent1'])    
    return mols, fracs

# %%
# InChI to SMILES
target_cols = ['inchi_solvent1',
               'inchi_solvent2',
                'inchi_solute']

all_inchis = pd.unique(df[target_cols].values.ravel())

smiles_map = {}

for inchi in tqdm(all_inchis):
    if pd.notna(inchi):
        smiles_map[inchi] = inchi_to_smiles(inchi)

for col in target_cols:
    df[f'{col}_smiles'] = df[col].map(smiles_map)
#%%
# Solute graph
df['solute_mol'] = df['inchi_solute_smiles'].apply(smiles_to_mol)

# Solvent graph
df['solvent1_mol'] = df['inchi_solvent1_smiles'].apply(smiles_to_mol)
df['solvent2_mol'] = df['inchi_solvent2_smiles'].apply(smiles_to_mol)

# %%
df['mols'], df['mol_fractions'] = zip(*df.apply(get_mol_list, axis=1))
# %%
processed_data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    sample = {
        'solv1_name': row['inchi_solvent1_smiles'],
        'solv2_name': row['inchi_solvent2_smiles'],
        'solu_name': row['inchi_solute_smiles'],
        'mols': row['mols'],
        'mol_fractions': row['mol_fractions'],
        'target': [row['Gsolv (kcal/mol)'],row['Hsolv (kcal/mol)']]
    }
    processed_data.append(sample)


# %% save the processed data
with open('train_binary.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("Data preprocessing completed. Total samples:", len(processed_data))

# %%
