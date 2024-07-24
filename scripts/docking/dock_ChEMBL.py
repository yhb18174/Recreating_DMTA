import pandas as pd
import sys

sys.path.insert(0, '/users/yhb18174/Recreating_DMTA/scripts/docking/')
from docking_fns import GNINA_fns, Run_MP_GNINA

df = pd.read_csv('/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/ChEMBL_docking_df.csv', index_col=False)

docking_dir = '/users/yhb18174/Recreating_DMTA/docking/ChEMBL/'
smi_ls = list(df['SMILES'])
molid_ls = list(df['ID'])

mp = Run_MP_GNINA(docking_dir=docking_dir,
                  molid_ls=molid_ls,
                  smi_ls=smi_ls,
                  receptor_path="/users/yhb18174/Recreating_DMTA/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt")

mp._make_sdfs()

ids, scores = mp._submit_jobs(max_time=1)

df['affinity_exp'] = scores

df.to_csv('/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/comp_ChEMBL_docking_df.csv', index='ID')