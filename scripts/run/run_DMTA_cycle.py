import pandas as pd
import multiprocessing as mp
import numpy as np
import pickle as pk
from pathlib import Path
import sys
import time
from datetime import datetime
import fcntl

FILE_DIR = str(Path(__file__).parent)
PROJ_DIR = str(Path(__file__).parent.parent.parent)

# Inserting paths to relevant functions
# Models
sys.path.insert(0, PROJ_DIR + '/models/')
# Docking
sys.path.insert(0, PROJ_DIR + '/docking/')
# Dataset
sys.path.insert(0, PROJ_DIR + '/dataset/')
# Molecule Selection
sys.path.insert(0, PROJ_DIR + '/mol_sel/')
# Misc
sys.path.insert(0, PROJ_DIR + '/misc/')

from RF_class import RF_Model
from misc_functions import molid2batchno
from dataset_fns import Dataset_Accessor
from docking_fns import GNINA_fns, Run_MP_GNINA

# Setting initial values
start_iter = 1
total_iters = 150
n_batches = 75
n_cpus = 40
n_cmpds = 10
mol_prefix = 'HW-'
rdkit_or_mordred = 'rdkit'

# Setting runtimes
time_ran = 0
max_sec_it = 60*60*4
max_runtime = 60*60*168
run_times = []
avg_runtime = np.mean(run_times)
max_it_runtime = avg_runtime + 60*60

# Setting selection method
sel_meth = ''
dataset_file = ''
desc_file = ''
docking_score_file = ''

# Directories (must end in '/')
docking_dir = PROJ_DIR + '/PyMolGen/'
results_dir = PROJ_DIR + f'/RF_model/results/{rdkit_or_mordred}_desc/'
run_dir = results_dir + sys.argv[1] + '/'
init_model_dir = run_dir + 'init_RF_model/it0/'

# Receptor
receptor_path = ''


# ============== #
# Run iterations #
# ============== #
if n_cpus == -1:
    n_cpus = mp.cpu_count()
print(f'Running with {n_cpus} CPUs')

it_ran_ls = []

# Start iterations loop
for iter in range(start_iter, start_iter+total_iters):
    if time_ran + max_it_runtime < max_runtime:
        it_start_time= time.time()

        print(f'\n+===========Iteration: {iter}===========+')

        it_dir= run_dir+'it'+str(iter)+'_running/'

        if iter-1 == 0:
            prev_it_dir = init_model_dir
        else:
            prev_it_dir= run_dir+'it'+str(iter-1)+'/'

        dir=Path(it_dir)

        if not dir.exists():
            dir.mkdir()

        # ---------------------------- #
        # Select next set of compounds #
        # ---------------------------- #
        print('- Selecting next set of compounds...')

        if sel_meth == 'r':
            sel_idx = ''
            print('random')

        elif sel_meth == 'mp':
            sel_idx = ''
            print('most potent')

        elif sel_meth == 'mu':
            sel_idx = ''
            print('most uncertain')

        elif sel_meth == 'mpo':
            sel_idx = ''
            print('most mpo')

        elif sel_meth == 'rmp':
            sel_idx = ''
            print('random most potent')

        elif sel_meth == 'rmpo':
            sel_idx = ''
            print('random most mpo')

        df_select = pd.DataFrame(data=[], columns=[], index=sel_idx)
        df_select.index.rename('ID', inplace=True)

        # Figuring out which batch data the molecules are in
        # First need to strip the prefix from the IDs
        
        df_select['batch_no'] = [molid2batchno(molid) for molid in df_select.index]

        # ------------------------------------------ #
        # Read selection and run docking if required #
        # ------------------------------------------ #

        start_t = datetime.now()
        print('- Running docking if required...')
        print('  {:>16s}\t{:>15s}'.format('ID', 'Docking score'))
        print('  {:>16s}\t{:>15s}'.format('----------------', '---------------'))

        # Read selected compounds, grouped by batch:
        for batch_no, idxs_in_batch in df_select.reset_index()\
                                                .grou4pby('batch_no')['ID']\
                                                .apply(list)\
                                                .items():
            
            print(f'Docking molecules in batch {batch_no}...')
            
            # Read csv file containing docking scores to see if compounds have already been docked
            docking_score_batch_file = docking_score_file.replace('?', str(batch_no))
            
            # Obtain exclusive acess to the file
            da = Dataset_Accessor(original_path=docking_score_batch_file,
                                temp_suffix='.dock',
                                wait_time=30)
            
            docking_file = da.get_exclusive_access()
        
            if docking_file is None:
                print(f'Failed to access0 file:\n{docking_score_batch_file}')
                print(f'Redocking of IDs:\n{idxs_in_batch} required')
                continue

            dock_df = pd.read_csv(docking_file, index_col=0)

            # Indices of compounds for docking:
            for_docking = dock_df.loc[idxs_in_batch]\
                                .loc[dock_df['Docking_Score'] == 'N/A', []]
            
            # Change affinity_dock value for each molecule being docked as 'PD' (pending)   
            dock_df.loc[for_docking.index, 'affinity_dock'] = 'PD'
            da.edit_df(column_to_edit='affinity_dock',
                        idx_to_edit=idxs_in_batch,
                        vals_to_enter=['PD' for idx in idxs_in_batch])
            
            # Release exclusive access on the File so other parallel runs can access it    
            da.release_file()
                
            print('** Docking compounds: ' +', '.join(for_docking.index.tolist()), end='\r')

            dock_scores_ls = []

            molid_ls = []
            smi_ls = []
            for mol_id, smi in for_docking['SMILES'].items():
                molid_ls.append(mol_id)
                smi_ls.append(smi)

            docker = Run_MP_GNINA(docking_dir=docking_dir,
                                molid_ls=molid_ls,
                                smi_ls=smi_ls,
                                receptor_path=receptor_path)
        
            docker._make_sdfs()


            for_docking['affinity_dock'] = dock_scores_ls
            da.get_exclusive_access()

            da.edit_df(column_to_edit='affinity_dock',
                    idx_to_edit=idxs_in_batch,
                    vals_to_enter=dock_scores_ls)
            
            da.release_file()

            print('\n')
            
            for mol_id in idxs_in_batch:
                if dock_df.loc[mol_id, 'affinity_dock'] != 'PD':
                    print('  {:>16s}\t{:>15s}\t{:>15s}'.format(
                        str(mol_id), str(dock_df.loc[mol_id, 'Docking_score'])))
                    
            it_fin_time = time.time()
            time_ran += it_fin_time - it_start_time
            it_ran_ls.append(iter)

        else:
            print(f'Ran out of time to run another full iteration.')
            break
    
    print('Iterations ran:\n{it_ran_ls}')