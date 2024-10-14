import pandas as pd
import multiprocessing as mp
import numpy as np
import pickle as pk
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
# Dataset
sys.path.insert(0, PROJ_DIR + "/scripts/dataset/")
# Molecule Selection
sys.path.insert(0, PROJ_DIR + "/scripts/mol_sel/")
# Misc
sys.path.insert(0, PROJ_DIR + "/scripts/misc/")

from misc_functions import molid2batchno
from dataset_functions import Dataset_Accessor
from docking_fns import Run_GNINA, WaitForDocking, GetUndocked
from mol_sel_fns import Molecule_Selector

# Setting initial values
start_iter = 1
total_iters = 2
n_batches = 75
n_cpus = 40
n_cmpds = 10
mol_prefix = "HW-"
rdkit_or_mordred = "rdkit"
docking_column = "CNN_Affinity"

# Setting runtimes
time_ran = 0
max_sec_it = 60 * 60 * 4
max_runtime = 60 * 60 * 168
run_times = []
avg_runtime = np.mean(run_times)
max_it_runtime = avg_runtime + 60 * 60

sel_meth = "r"


# Directories (must end in '/')
docking_dir = PROJ_DIR + "/docking/PyMolGen/"
results_dir = PROJ_DIR + f"/results/{rdkit_or_mordred}_desc/"
run_dir = results_dir + sys.argv[1] + "/"
init_model_dir = run_dir + "init_RF_model/it0/"
test_dir = PROJ_DIR + "/datsets/test_data/"
dataset_file = test_dir + "PMG_rdkit_full_batch_1.csv.gz"
desc_file = test_dir + "PMG_rdkit_desc_batch_1.csv.gz"
docking_score_file = test_dir + "PMG_docking_?.csv.gz"
chosen_mol_file = run_dir + "chosen_mol.csv"

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
    if time_ran + max_it_runtime < max_runtime:
        it_start_time = time.time()

        print(f"\n+===========Iteration: {iter}===========+")

        it_dir = run_dir + "it" + str(iter) + "_running/"

        if iter - 1 == 0:
            prev_it_dir = init_model_dir
        else:
            prev_it_dir = run_dir
        # ---------------------------- #
        # Select next set of compounds #
        # ---------------------------- #
        print("- Selecting next set of compounds...")

        selector = Molecule_Selector(
            n_cmpds=n_cmpds,
            preds_dir=run_dir,
            chosen_mol_file=chosen_mol_file,
            iteration=iter,
        )
        if sel_meth == "r":
            sel_idx = selector.random()

        elif sel_meth == "mp":
            sel_idx = selector.best(column="Affinity_pred", ascending=False)

        elif sel_meth == "mu":
            sel_idx = selector.best(column="Uncertainty", ascending=False)

        elif sel_meth == "mpo":
            sel_idx = selector.best(column="MPO", ascending=True)

        elif sel_meth == "rmp":
            sel_idx = selector.random_in_best(
                column="Affinity_pred", ascending=False, frac=0.1
            )

        elif sel_meth == "rmpo":
            sel_idx = selector.random_in_best(column="MPO", ascending=True, frac=0.1)

        df_select = pd.DataFrame(data=[], columns=[], index=sel_idx)
        df_select.index.rename("ID", inplace=True)

        # Figuring out which batch data the molecules are in
        # First need to strip the prefix from the IDs

        df_select["batch_no"] = [molid2batchno(molid) for molid in df_select.index]

        # ------------------------------------------ #
        # Read selection and run docking if required #
        # ------------------------------------------ #

        start_t = datetime.now()
        print("- Running docking if required...")
        print("  {:>16s}\t{:>15s}".format("ID", "Docking score"))
        print("  {:>16s}\t{:>15s}".format("----------------", "---------------"))

        # Read selected compounds, grouped by batch:
        for batch_no, idxs_in_batch in (
            df_select.reset_index().groupby("batch_no")["ID"].apply(list).items()
        ):

            print(f"Docking molecules in batch {batch_no}...")

            # Read csv file containing docking scores to see if compounds have already been docked
            docking_score_batch_file = docking_score_file.replace("?", str(batch_no))

            # Obtain exclusive acess to the file
            da = Dataset_Accessor(
                original_path=docking_score_batch_file,
                temp_suffix=".dock",
                wait_time=30,
            )

            docking_file = da.get_exclusive_access()

            if docking_file is None:
                print(f"Failed to access file:\n{docking_score_batch_file}")
                print(f"Redocking of IDs:\n{idxs_in_batch} required")
                continue

            dock_df = pd.read_csv(docking_file, index_col=0)

            # Indices of compounds for docking:
            for_docking = GetUndocked(dock_df=dock_df, idxs_in_batch=idxs_in_batch)

            # Change affinity_dock value for each molecule being docked as 'PD' (pending)
            da.edit_df(
                column_to_edit=docking_column,
                idx_to_edit=idxs_in_batch,
                vals_to_enter=["PD" for idx in idxs_in_batch],
            )

            # Release exclusive access on the File so other parallel runs can access it
            da.release_file()

            print(
                "** Docking compounds: " + ", ".join(for_docking.index.tolist()),
                end="\r",
            )

            molid_ls = []
            smi_ls = []

            for mol_id, smi in for_docking["SMILES"].items():
                molid_ls.append(mol_id)
                smi_ls.append(smi)

            docker = Run_GNINA(
                docking_dir=docking_dir,
                molid_ls=molid_ls,
                smi_ls=smi_ls,
                receptor_path=receptor_path,
            )

            docker._make_ph74_sdfs()
            dock_scores_ls = docker._submit_jobs(max_time=2)

            for_docking[docking_column] = dock_scores_ls
            da.get_exclusive_access()

            da.edit_df(
                column_to_edit=docking_column,
                idx_to_edit=idxs_in_batch,
                vals_to_enter=dock_scores_ls,
            )

            da.release_file()

            print("\n")

            for mol_id in idxs_in_batch:
                if dock_df.loc[mol_id, docking_column] != "PD":
                    print(
                        "  {:>16s}\t{:>15s}\t{:>15s}".format(
                            str(mol_id), str(dock_df.loc[mol_id, "Docking_score"])
                        )
                    )

            WaitForDocking(
                dock_df,
                idxs_in_batch=idxs_in_batch,
            )

            it_fin_time = time.time()
            time_ran += it_fin_time - it_start_time
            it_ran_ls.append(iter)

        else:
            print(f"Ran out of time to run another full iteration.")
            break

    print("Iterations ran:\n{it_ran_ls}")
