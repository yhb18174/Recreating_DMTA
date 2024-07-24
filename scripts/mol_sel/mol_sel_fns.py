import pandas as pd
import numpy as np
from glob import glob
import random

class Molecule_Selector():
    def __init__(self,
                 n_cmpds: int,
                 preds_dir: str,
                 chosen_mol_file: str):
        
        self.n_cmpds = n_cmpds
        self.preds_files = glob(f'{preds_dir}all_preds*')
        self.chosen_mol = pd.read_csv(chosen_mol_file, compression='zip')

    def random(self,
               preds_dir: str):

        mols = []
        
        while len(mols) < self.n_cmpds:
            file = random.choice(self.preds_files)
            df = pd.read_csv(file, compression='gzip')
            id = random.choice(df['ID']
            if id not in list(self.chosen_mol['ID']):
                mols.append(id)
            else:
                pass    
                
        return mols
    
    def best(self,
             column: str,
             ascending: bool):
        
        mols = []
        top_df_ls = [pd.read_csv(preds_file).sort_values(by=column, ascending=ascending).head(1000) for preds_file in self.preds_files]
        full_df = pd.concat(top_df_ls).sort_values(by=column, ascending=ascending)

        while len(mols) < self.n_compounds:
            for id in full_df['ID']:
                if id not in list(self.chosen_mol['ID']):
                    mols.append(id)
                else:
                    pass
        
        return mols

    def random_in_best(self,
                       column,
                       ascending,
                       frac):

        mols = []
        total_mols = 0
        df_ls = [pd.read_csv(preds_file).sort_values(by=column, ascending=ascending) for preds_file in self.preds_files]

        for dfs in df_ls:
            total_mols += len(dfs)
        
        n_mols = int(total_mols*frac)

        top_df_ls = [df.sort_values(by=column, ascending=ascending).head(n_mols). for dfs in df_ls]
        full_df = pd.concat(top_df_ls).sort_values(by=column, ascending=ascending).head(total_mols)

        while len(mols) < self.n_compounds:
            id = random.choice(full_df['ID'])
            if id not in list(self.chosen_mol['ID'])
        

