import pandas as pd
import numpy as np
from glob import glob
import random
from pathlib import Path


class Molecule_Selector:
    """
    Description
    -----------
    A class which holds all of the molecular selection methods:
    • Random selection of molecules
    • Highest/lowest value based on a provided column
    • Random choice from the highest/lowest value
    """

    def __init__(
        self,
        n_cmpds: int,
        preds_dir: str,
        chosen_mol_file: str,
        iteration: int,
        all_preds_prefix: str = "all_preds",
    ):
        """ "
        Description
        -----------
        Initialising the class to globally hold the number of compounds, the prediction files and
        chosen_mol file. This will chose molecules not previously chosen by the algorithm, as
        indicated by compounds in the chosen_mol file.

        Parameters
        ----------
        n_cmpds (int)           Number of compounds to chose during the selection process
        preds_dir (str)         Directory containing all of the prediction files
        chosen_mol_file (str)   .csv file containing the IDs of previously chosen molecules
        iteration (int)         Iteration which the molecule selector is being called on
                                (used to update chosen_mol file)
        all_preds_prefix (str)  Prefix of the all_preds files which files have in common
                                (all_preds by default)

        Returns
        -------
        Class-wide/global instances of n_cmpds, preds_files, chosen_mol file, and iteration number

        """

        self.n_cmpds = n_cmpds
        self.preds_files = glob(f"{preds_dir}{all_preds_prefix}*")
        self.chosen_mol_file = chosen_mol_file

        if not Path(self.chosen_mol_file).exists():
            self.chosen_mol = pd.DataFrame(columns=["Iteration"])
            self.chosen_mol.index.name = "ID"
            self.chosen_mol.to_csv(self.chosen_mol_file, index_label="ID")
        else:
            self.chosen_mol = pd.read_csv(chosen_mol_file, index_col="ID")

        self.it = iteration

    def random(self):
        """
        Description
        -----------
        Function to select molecules at random across all of the prediction files

        Parameters
        ----------
        None

        Returns
        -------
        List of randomly chosen molecule IDs which are not in the chosen_mol file
        """

        mols = []

        while len(mols) < self.n_cmpds:
            file = random.choice(self.preds_files)
            df = pd.read_csv(file, index_col="ID", compression="gzip")
            id = random.choice(df.index.tolist())
            if id not in list(self.chosen_mol.index):
                mols.append(id)
            else:
                pass

        self.update_chosen_mol(mols)

        return mols

    def best(self, column: str, ascending: bool):
        """
        Description
        -----------
        Function to select the top molecules based off of a selection criteria in a .csv file , e.g., predictive uncertainty.

        Parameters
        ----------
        column (str)        Name of column to sort molecules by
        ascending (bool)    Flag to swap how the molecules are sorted.
                            True = Lowest to highest (top to bottom)
                            False = Highest to lowest (top to bottom)

        Returns
        -------
        List of best molecules chosen based off of defined column which are not in the chosen_mol file
        """

        mols = []
        top_df_ls = [
            pd.read_csv(preds_file, index_col="ID", compression="gzip")
            .sort_values(by=column, ascending=ascending)
            .head(1000)
            for preds_file in self.preds_files
        ]
        full_df = pd.concat(top_df_ls).sort_values(by=column, ascending=ascending)

        for id in full_df.index:
            if id not in list(self.chosen_mol.index):
                mols.append(id)
            if len(mols) >= self.n_cmpds:
                break

        self.update_chosen_mol(mols)

        return mols

    def random_in_best(self, column: str, ascending: bool, frac: float):
        """
        Description
        -----------
        Function to choose molecules at random from the top % of molecules, e.g., randomly choosing moleucles within the top
        10 % of predicted potency.

        Parameters
        ----------
        column (str)        Name of column to sort molecules by
        ascending (bool)    Flag to swap how the molecules are sorted.
                            True = Lowest to highest (top to bottom)
                            False = Highest to lowest (top to bottom)
        frac (float)        Fraction of data to look over, (0.1 = 10 %, etc.)

        Returns
        -------
        List of molecules chosen at random within the top % of molecules. Choses molecules not already present in chosen_mol file
        """
        mols = []
        total_mols = 0
        df_ls = [
            pd.read_csv(preds_file, index_col="ID", compression="gzip").sort_values(
                by=column, ascending=ascending
            )
            for preds_file in self.preds_files
        ]

        for dfs in df_ls:
            total_mols += len(dfs)

        n_mols = int(total_mols * frac)

        top_df_ls = [
            df.sort_values(by=column, ascending=ascending).head(n_mols) for df in df_ls
        ]
        full_df = (
            pd.concat(top_df_ls)
            .sort_values(by=column, ascending=ascending)
            .head(total_mols)
        )

        while len(mols) < self.n_cmpds:
            id = random.choice(full_df.index.tolist())
            if id not in list(self.chosen_mol.index):
                mols.append(id)
            else:
                pass

        self.update_chosen_mol(mols)

        return mols

    def update_chosen_mol(self, mol_ls: list, save: bool = True):
        """
        Description
        -----------
        Function to update and save the chosen_mol file with molecules chosen by the given selection method

        Parameters
        ----------
        mol_ls (list)       List of molecules to enter into the chosen_mol file
        save (bool)         Flag to save the new chosen mol file

        Returns
        -------
        Newly updated chosen_mol file
        """

        new_rows = {mol: self.it for mol in mol_ls}

        for key, value in new_rows.items():
            self.chosen_mol.loc[key] = value

        if save:
            self.chosen_mol.to_csv(self.chosen_mol_file, index="ID")
        else:
            return self.chosen_mol
