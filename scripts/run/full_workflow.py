import pandas as pd
import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime


FILE_DIR = str(Path(__file__).parent)
PROJ_DIR = str(Path(__file__).parent.parent.parent)

# Inserting paths to relevant functions
# Models
sys.path.insert(0, PROJ_DIR + "/scripts/models/")
from RF_class import RF_model

# Docking
sys.path.insert(0, PROJ_DIR + "/scripts/docking/")
from docking_fns import Run_GNINA, WaitForDocking, GetUndocked

# Dataset
sys.path.insert(0, PROJ_DIR + "/scripts/dataset/")
from dataset_functions import Dataset_Accessor

# Molecule Selection
sys.path.insert(0, PROJ_DIR + "/scripts/mol_sel/")
from mol_sel_fns import Molecule_Selector

# Misc
sys.path.insert(0, PROJ_DIR + "/scripts/misc/")
from misc_functions import molid2batchno

# Setting initial values
start_iter = 1
total_iters = 1
n_batches = 75
n_cpus = 40
n_cmpds = 10
mol_prefix = "PMG-"
rdkit_or_mordred = "rdkit"
docking_column = 'CNN_yffinity'

# Setting runtimes
time_ran = 0
max_sec_it = 60 * 60 * 4
max_runtime = 60 * 60 * 168
run_times = []
avg_runtime = 0
max_it_runtime = avg_runtime + 60 * 60

sel_meth = "rmp"

# Directories (must end in '/')
docking_dir = PROJ_DIR + "/docking/PyMolGen/"
results_dir = PROJ_DIR + f"/results/test/"
run_dir = results_dir + 'test_1/'
init_model_dir = run_dir + "it0/"
test_dir = PROJ_DIR + '/datsets/test_data/'
dataset_file = test_dir + "PMG_rdkit_1.csv.gz"
desc_file = test_dir + "PMG_rdkit_desc_1.csv.gz"
docking_score_file = test_dir + "PMG_docking_1.csv.gz"
chosen_mol_file = run_dir + 'chosen_mol.csv'


# Receptor
receptor_path = PROJ_DIR + "/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt"

# ============== #
# Run iterations #
# ============== #
if n_cpus == -1:
    n_cpus = mp.cpu_count()
print(f"Running with {n_cpus} CPUs")

it_ran_ls = []

# Start iterations loop
for iter in range(start_iter, start_iter + total_iters):
    print(f"============= Running Iteration {iter} =============")
    if time_ran + max_it_runtime < max_runtime:
        it_start_time = time.time()
        
        it_dir = run_dir + "it" + str(iter) + "_running/"
        it_dir = Path(it_dir)
        it_dir.mkdir(exist_ok=True)

        if iter - 1 == 0:
            prev_it_dir = init_model_dir

        else:
            prev_it_dir = run_dir + f'it{iter-1}'

# ============================ #
# Select next set of compounds #
# ============================ #
            
        print("Selecting next set of compounds...")

        selector = Molecule_Selector(
            n_cmpds=n_cmpds,
            preds_dir=prev_it_dir,
            chosen_mol_file=chosen_mol_file,
            iteration=iter,
        )

        print(selector.preds_files)

        if sel_meth == "r":
            sel_idx = selector.random()

        elif sel_meth == "mp":
            sel_idx = selector.best(
                column="affinity_pred", ascending=False
            )

        elif sel_meth == "mu":
            sel_idx = selector.best(
                column="Uncertainty", ascending=False
            )

        elif sel_meth == "mpo":
            sel_idx = selector.best(
                column="MPO", ascending=True
            )

        elif sel_meth == "rmp":
            sel_idx = selector.random_in_best(
                column="affinity_pred", ascending=False, frac=0.1
            )

        elif sel_meth == "rmpo":
            sel_idx = selector.random_in_best(
                column="MPO", ascending=True, frac=0.1
            )

        df_select = pd.DataFrame(data=[], columns=[], index=sel_idx)
        df_select.index.rename("ID", inplace=True)

        df_select["batch_no"] = [molid2batchno(molid, mol_prefix, 'all_mols') for molid in df_select.index]

        print(df_select)
