import pandas as pd
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
import re
from prettytable import PrettyTable
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import sys
import time

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


def read_csv_files(filename, columns=None):
    return pd.read_csv(filename, compression="gzip")


def order_glob_files(file_ls: list):
    sorted_ls = sorted(
        file_ls, key=lambda x: int(re.findall(r"\d+", x.split("_")[-1])[0])
    )
    return sorted_ls


class RecDMTA:
    def __init__(
        self,
        full_data_fpath: str,
        full_data_fprefix: str,
        desc_fpath: str,
        desc_fprefix: str,
        start_iter: int,
        total_iters: int,
        n_cmpds: int,
        docking_dir: str,
        docking_file_dir: str,
        results_dir: str,
        init_model_dir: str,
        chosen_mol_file: str,
        selection_method: str,
        docking_score_files: str,
        run_name: str,
        docking_column,
        max_confs: int = 1,
        id_prefix: str = "PMG-",
        n_cpus: int = 40,
        receptor_path: str = PROJ_DIR
        + "/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt",
        max_runtime: int = 60 * 60 * 168,
        max_it_runtime: int = 60 * 60 * 10,
        hyper_params: dict = {
            "rf__n_estimators": [400, 500],
            "rf__max_features": ["sqrt"],
            "rf__max_depth": [25, 50, 75, 100],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [2, 4, 8],
        },
    ):
        """
        Description
        -----------
        Class to run the full RecDMTA project

        Parameters
        ----------
        full_data_fpath (str)       Path to the full data from each dataset (crucially includes
                                    the PFI and oe_logp columns)
        full_data_fprefix (str)     Prefix for the full_data csv files e.g., PMG_rdkit_*
        desc_fpath (str)            Path to the descriptors set for the data
        desc_fprefix (str)          Prefix for the descriptor csv files e.g., PMG_rdkit_desc_*
        start_iter (int)            Iteration to start the runs on
        total_iters (int)           Total number of iterations to complete
        n_cmpds (int)               Number of compounds chosen to dock & add to the training set
        docking_dir (str)           Path to the directory containing all GNINA outputs
        results_dir (str)           Path to the drectory containing all results
        init_model_dir (str)        Path to the initial model directory containing the predictions and
                                    initially trained model
        chosen_mol_file (str)       Path to the chosen_mol file, if none present enter where you'd like 
                                    the chosen mol file to be (end with the csv name too)
        selection_method (str)      Method used to select molecules for retraining:
                                            'r'     = random
                                            'mp'    = most potent
                                            'rmpo'  = random in most potent
                                            'mpo'   = lowest mpo
                                            'rmpo'  = random in lowest mpo
                                            'mu'    = most uncertain
        docking_score_files (str)   Docking score files with an '*' as a batch number replacement
                                    e.g., PMG_docking_*.csv
        id_prefix: (str)            Prefix to molecule IDs
        docking_column (str)        Column in the docking files which contain the docking infomation
        n_cpus (int)                Number of CPUs to use during the job
        receptor_path (str)         Pathway to the receptor used for docking
        max_runtime (int)           Maximum time a job can be run
        max_it_runtime (int)        Maximim tim an iteration can be run

        Returns
        -------
        Initialised class
        """

        global PROJ_DIR

        self.full_df = pd.DataFrame()

        if full_data_fpath.endswith("/"):
            self.full_fpath = full_data_fpath + full_data_fprefix
        else:
            self.full_fpath = full_data_fpath + "/" + full_data_fprefix

        if desc_fpath.endswith("/"):
            self.desc_fpath = desc_fpath + desc_fprefix
        else:
            self.desc_fpath = desc_fpath + "/" + desc_fprefix

        # Pathways
        self.full_flist = glob(f"{self.full_fpath}")
        self.full_flist = order_glob_files(self.full_flist)

        self.desc_flist = glob(f"{self.desc_fpath}")
        self.desc_flist = order_glob_files(self.desc_flist)

        self.docking_dir = docking_dir
        self.results_dir = results_dir
        self.init_model_dir = init_model_dir
        self.chosen_mol_file = chosen_mol_file
        self.receptor_path = receptor_path
        self.docking_score_files = docking_file_dir + docking_score_files
        self.PROJ_DIR = PROJ_DIR
        self.prev_it_dir = init_model_dir
        self.run_dir = self.results_dir + run_name + "/"
        mk_run_dir = Path(self.run_dir)
        if not mk_run_dir.exists():
            mk_run_dir.mkdir()
            link_name = mk_run_dir / "it0"
            link_name.symlink_to(Path(init_model_dir))
            print(f"Symbolic link created: \n{link_name} -> {Path(init_model_dir)}")

        # Iteration Settings
        self.start_iter = start_iter
        self.total_iters = total_iters
        self.n_cmpds = n_cmpds
        self.docking_column = docking_column
        self.n_cpus = n_cpus
        self.selection_method = selection_method
        self.id_prefix = id_prefix
        self.hyper_params = hyper_params
        self.max_confs = max_confs

        # Timings
        self.time_ran = 0
        self.max_runtime = max_runtime
        self.max_it_runtime = max_it_runtime
        self.run_times = []
        if self.run_times:
            self.avg_runtime = np.mean(self.run_times)
        else:
            self.avg_runtime = 0

    def SelectCmpds(self, iteration: int, prev_it_dir: str, frac: float = 0.1):
        """
        Description
        -----------
        Function to select molecules from a the previous iterations directory and add them to the chosen_mol.csv
        file. Also creates a pd.DataFrame with the selected molecules, and the batch number they belong to.
        
        Parameters
        ----------
        iteration (int)         Iteration which you are on, this saves which iteration each molecule was selected
                                in the chosen_mol file
        prev_it_dir (str)       Pathway to the precious iteration directory to select the molecules from the 
                                prediction files
        frac (float)            Fraction of the dataset to choose within when selecting on a portion of the data

        Returns
        -------
        A pandas.DataFrame object with the following structure:
                        ____ __________
                        | ID | batch_no |
                        |----|----------|
                        | x  |    1     |
                        |____|__________|
        """

        self.sel = Molecule_Selector(
            n_cmpds=self.n_cmpds,
            preds_dir=prev_it_dir,
            chosen_mol_file=self.chosen_mol_file,
            iteration=iteration,
        )

        if self.selection_method == "r":
            sel_idx = self.sel.random()

        elif self.selection_method == "mp":
            sel_idx = self.sel.best(column="affinity_pred", ascending=False)

        elif self.selection_method == "rmp":
            sel_idx = self.sel.random_in_best(
                column="affinity_pred", ascending=False, frac=frac
            )

        elif self.selection_method == "mu":
            sel_idx = self.sel.best(column="Uncertainty", ascending=False)

        elif self.selection_method == "mpo":
            sel_idx = self.sel.best(column="MPO", ascending=True)

        elif self.selection_method == "rmpo":
            sel_idx = self.sel.random_in_best(column="MPO", ascending=True, frac=frac)

        elif self.selection_method == "test":
            sel_idx = ["PMG-31895", "PMG-27063"]

        self.df_select = pd.DataFrame(data=[], columns=[], index=sel_idx)
        self.df_select.index.rename("ID", inplace=True)
        self.df_select["batch_no"] = [
            molid2batchno(molid, self.id_prefix, prev_it_dir + "all_preds*")
            for molid in self.df_select.index
        ]

        table = PrettyTable()
        table.field_names = ["ID"] + list(self.df_select.columns)
        for row in self.df_select.itertuples(index=True):
            table.add_row(row)

        print(table)

        return self.df_select
    
    def _run_docking_wrapper(self, args):
        batch_no, idxs_in_batch = args

        docking_score_batch_file = self.docking_score_files.replace(
               "*", str(batch_no)
            )
        
        da = Dataset_Accessor(
        original_path=docking_score_batch_file,
        temp_suffix=".dock",
        wait_time=30,
            )
        
        # Obtain exclusive access to the docking file
        docking_file = da.get_exclusive_access() 
        if docking_file is None:
            print(f"Failed to access file:\n{docking_score_batch_file}")
            print(f"Redocking of IDs:\n{idxs_in_batch} required")
            return

        dock_df = pd.read_csv(docking_file, index_col=0)

        # Isolating the molecule ids which have not already been docked or in the process of being docked
        for_docking = GetUndocked(
            dock_df=dock_df,
            idxs_in_batch=idxs_in_batch,
            scores_col=self.docking_column,
        )

        if for_docking.empty:
            print(f"No molecules to dock in batch {batch_no}...")
            da.release_file()
            batch_dock_df = pd.read_csv(docking_score_batch_file, index_col=0).loc[
                idxs_in_batch
            ]
            return batch_dock_df

        # Change docking value fr each molecule being docked as 'PD' (pending)
        da.edit_df(
            column_to_edit=self.docking_column,
            idxs_to_edit=for_docking.index,
            vals_to_enter=["PD" for idx in for_docking.index],
        )

        # Releases exclusive access on file so parallel runs can access it
        da.release_file()

        print(
            "** Docking compounds: " + ", ".join(for_docking.index.tolist()),
            end="\r",
        )

        molid_ls = []
        smi_ls = []

        for molid, smi in for_docking["SMILES"].items():
            molid_ls.append(molid)
            smi_ls.append(smi)

        # Initialising the docker
        docker = Run_GNINA(
            docking_dir=self.docking_dir,
            molid_ls=molid_ls,
            smi_ls=smi_ls,
            receptor_path=self.receptor_path,
            max_confs=self.max_confs,
        )

        # Creating sdfs with numerous conformers and adjusting for pH 7.4
        docker.ProcessMols(use_multiprocessing=False)

        # Docking the molecules and saving scores in for_docking
        dock_scores_ls = docker.SubmitJobs(run_hrs=0,
                                           run_mins=20,
                                           use_multiprocessing=False)

        if self.docking_column == "CNN_affinity":
            n = 1
        else:
            n = 2

        for_docking[self.docking_column] = dock_scores_ls[n]

        da.get_exclusive_access()

        da.edit_df(
            column_to_edit=self.docking_column,
            idxs_to_edit=molid_ls,
            vals_to_enter=dock_scores_ls[n],
        )

        da.release_file()

        WaitForDocking(
            docking_score_batch_file,
            idxs_in_batch=idxs_in_batch,
            scores_col=self.docking_column,
        )

        batch_dock_df = pd.read_csv(docking_score_batch_file, index_col=0)
        batch_dock_df = batch_dock_df.loc[idxs_in_batch]
        
        return batch_dock_df


    def RunDocking(self,):
        """
        Description
        -----------
        Function to run the docking portion of the workflow

        Parameters
        ---------
        None

        Returns
        -------
        pd.DataFrame object containing the docking information for the ids selected
        """
        
        args = [
            (batch_no, idxs_in_batch)
            for batch_no, idxs_in_batch in (
                self.df_select.reset_index().groupby("batch_no")["ID"].apply(list).items()
            )
        ]

        with Pool(self.n_cpus) as pool:
            results = pool.map(self._run_docking_wrapper, args)

        results = [result for result in results if result is not None]

        self.fin_dock_df = pd.concat(results, axis=0)

        return self.fin_dock_df.loc[self.df_select.index]

    def UpdateTrainingSet(self,):
        """
        Description
        -----------
        Function to obtain the previous training data and ammend the new, chosen molecules to it
        
        Parameters
        ----------
        None
        
        Returns
        -------
        x2 pd.DataFrame objects:      1- The dataframe of the updated target values
                                      2- The dataframe of the updated feature set
        
        """

        # Obtaining the training data from the previous iteration
        training_dir = self.prev_it_dir + "/training_data/"

        prev_feats = pd.read_csv(
            training_dir + "training_features.csv.gz",
            index_col="ID",
            compression="gzip",
        )
        prev_targs = pd.read_csv(
            training_dir + "training_targets.csv.gz", index_col="ID", compression="gzip"
        )

        # Saving the columns which the previous iteration was trained on
        # (used to make sure the models are trained on the same set of features)
        self.prev_feat_cols = prev_feats.columns.tolist()

        # Dropping any molecules which failed to dock
        self.fin_dock_df = self.fin_dock_df.dropna(subset=[self.docking_column])

        # Getting the molecule IDs and their batch number to extract data from
        ids = self.fin_dock_df.index
        batch_nos = [
            molid2batchno(molid, self.id_prefix, self.desc_fpath) for molid in ids
        ]

        batch_to_ids = defaultdict(list)

        for id, batch_no in zip(ids, batch_nos):
            batch_to_ids[batch_no].append(id)

        added_desc = pd.DataFrame(columns=prev_feats.columns)

        # Creating a new df with all of the new data needed
        for batch in batch_to_ids:
            ids_in_batch = batch_to_ids[batch]
            desc_csv = self.desc_fpath.replace("*", str(batch))
            desc_df = pd.read_csv(
                desc_csv, index_col="ID", usecols=prev_feats.reset_index().columns
            )
            desc_df = desc_df.loc[ids_in_batch]
            added_desc = pd.concat([added_desc, desc_df], axis=0)

        # Adding new rows onto the previous training data sets
        self.updated_feats = pd.concat([prev_feats, added_desc], axis=0)
        self.updated_targs = pd.concat(
            [prev_targs, self.fin_dock_df[[self.docking_column]]], axis=0
        )

        return self.updated_targs, self.updated_feats

    def _predict_for_files(self, args: list):
        """
        Description
        -----------
        Function to wrap the predictive model in to allow for multiprocessing

        Parameters
        ----------
        args (list)     List containing the following parameters:
                        index        (number to give the suffix label the preds file)
                        desc_file    (descriptor file, given as a pathway)
                        full_file    (full file, given as a pathway)
                        model        (loaded prediction model class)

        Returns
        -------
        None
        """

        index, desc_file, full_file, model = args

        feats = pd.read_csv(desc_file, index_col="ID")

        model.Predict(
            feats=feats[self.prev_feat_cols],
            save_preds=True,
            preds_save_path=self.it_dir,
            preds_filename=f"all_preds_{index+1}",
            final_rf=self.it_dir + "/final_model.pkl",
            full_data_fpath=full_file,
        )

    def RetrainAndPredict(self):

        """
        Description
        -----------
        Function to firstly retrain a new model using the updated data from UpdateTrainingSet() function
        Then uses the trained to predict docking scores
    
        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Initialising the RF class
        model = RF_model(docking_column=self.docking_column)

        # Training on updated features
        rf = model.Train_Regressor(
            search_type="grid",
            hyper_params=self.hyper_params,
            features=self.updated_feats,
            targets=self.updated_targs,
            save_path=self.it_dir,
            save_final_model=True,
            plot_feat_importance=True,
        )

        self.it_rf_model = rf[0]

        # Setting up arguments for the _predict_for_files() function
        args = [
            (i, desc_file, full_file, model)
            for i, (desc_file, full_file) in enumerate(
                zip(self.desc_flist, self.full_flist)
            )
        ]

        # Multiprocessing through all full & descriptor files to make predictions
        with Pool(self.n_cpus) as pool:
            pool.map(self._predict_for_files, args)

        return self.it_rf_model

    def RunIterations(self, held_out_test_feats: str, held_out_test_targs: str):

        """
        Description
        -----------
        Function to run the full RecDMTA workflow

        Parameters
        ---------
        held_out_test_desc (str)     Path to csv with descriptors for held out data to make predictions on
        held_out_test_targs (str)    Path to csv with descriptors for held out docking scores to compare against predictions

        Returns
        -------
        None
        """
        if self.n_cpus == -1:
            self.n_cpus = mp.cpu_count()

        print(f"Running with {self.n_cpus} CPUs")

        it_ran_ls = []

        # Starting iterations loop
        for self.iter in range(self.start_iter, self.start_iter + self.total_iters):

            # Checking to see if the iteration will run over the total runtime allocation
            if self.time_ran + (self.avg_runtime * 1.5) < self.max_it_runtime:
                it_start_time = time.time()

                print(f"\n+===========Iteration: {self.iter}===========+\n")

                # Setting up the run directory, naming it _running/
                self.it_dir = self.run_dir + "it" + str(self.iter) + "_running/"
                mk_it_dir = Path(self.it_dir)
                mk_it_dir.mkdir(exist_ok=True)

                # Setting up the training_data directory
                mk_train_data = mk_it_dir / "training_data"
                mk_train_data.mkdir(exist_ok=True)

                if self.iter - 1 != 0:
                    self.prev_it_dir = self.run_dir + "it" + str(self.iter - 1) + "/"

                self.SelectCmpds(iteration=self.iter, prev_it_dir=self.prev_it_dir)

                self.RunDocking()

                self.UpdateTrainingSet()

                self.RetrainAndPredict()

                # Renaming iteration directory
                rename_it_dir = Path(self.it_dir)
                rename_it_dir.rename(Path(self.run_dir) / f"it{self.iter}")

                # Predict on held out test set
                cols = self.updated_feats.columns

                test_targs = pd.read_csv(held_out_test_targs, index_col="ID")
                test_targs = test_targs[test_targs[self.docking_column] != "False"][
                    self.docking_column
                ]
                test_feats = pd.read_csv(held_out_test_feats, index_col="ID")
                test_feats = test_feats[cols].loc[test_targs.index]
                model = RF_model(docking_column=self.docking_column)

                bias, sdep, mse, rmse, r2, true, pred = model._calculate_performance(
                    feature_test=test_feats,
                    target_test=test_targs,
                    best_rf=self.it_rf_model,
                )
                stats = pd.DataFrame(
                    {"Value": [bias, sdep, mse, rmse, r2]},
                    index=["Bias", "SDEP", "MSE", "RMSE", "r2"],
                )

                held_out_dir = Path(self.run_dir + f"/it{self.iter}/held_out_test/")
                held_out_dir.mkdir(exist_ok=True)

                stats.to_csv(str(held_out_dir) + "/held_out_test_stats.csv")

                pred_df = pd.DataFrame(
                    index=test_feats.index,
                    data=pred,
                    columns=["pred_Affinity(kcal/mol)"],
                )
                pred_df.index.name = "ID"
                pred_df.to_csv(str(held_out_dir) + "/held_out_test_preds.csv")

                # Calculating timings
                it_fin_time = time.time()
                iter_time = it_fin_time - it_start_time
                self.run_times.append(iter_time)
                self.time_ran += iter_time
                it_ran_ls.append(self.iter)

                print(f"\n+=========== Iteration Completed: {self.iter} ===========+")
                print(f"Iteration Run Time: {round(iter_time, 1)}")
                print(f"Average Iteration Run Time: {round(self.avg_runtime, 1)}")

        print(f"Iterations Ran:\n{it_ran_ls}")
