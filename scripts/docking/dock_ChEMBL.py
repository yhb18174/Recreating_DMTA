import pandas as pd
import sys

sys.path.insert(0, '/users/yhb18174/Recreating_DMTA/scripts/docking/')
from docking_fns import Run_GNINA

df = pd.read_csv('/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/ChEMBL_docking_smi.csv', index_col=False)

docking_dir = '/users/yhb18174/Recreating_DMTA/docking/new_ChEMBL/'
smi_ls = list(df['SMILES'])[:10]
molid_ls = list(df['ID'])[:10]

mp = Run_GNINA(docking_dir=docking_dir,
                  molid_ls=molid_ls,
                  smi_ls=smi_ls,
                  receptor_path="/users/yhb18174/Recreating_DMTA/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt")

mp.ProcessMols(use_multiprocessing=True)

ids, scores = mp.SubmitJobs(max_time=1)

df['CNN_affinity'] = scores

df.to_csv('/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/new_ChEMBL_docking_df.csv', index='ID')

mp.MakeCsv(save_data=True)

mp.CompressFiles()
