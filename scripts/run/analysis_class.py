import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from glob import glob
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import subprocess
from functools import partial
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from rdkit.DataStructs import FingerprintSimilarity
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import colorcet as cc

FILE_DIR = Path(__file__).parent
PROJ_DIR = Path(__file__).parent.parent.parent

# Misc
sys.path.insert(0, str(PROJ_DIR) + "/scripts/misc/")
from misc_functions import (
    molid2batchno,
    count_number_iters,
    count_conformations,
    get_chembl_molid_smi,
    get_sel_mols_between_iters,
    molid_ls_to_smiles,
)


def CalcStats(
    df: pd.DataFrame,
    true_col: str = "Affinity(kcal/mol)",
    pred_col: str = "pred_Affinity(kcal/mol)",
):

    # Extract True and Predicted Values from DF
    true = df[true_col].values
    pred = df[pred_col].values

    # Pearsons r^2
    pearson_r, _ = pearsonr(true, pred)
    pearson_r2 = pearson_r**2

    # Coefficient of Determination (R^2)
    cod = r2_score(true, pred)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(true, pred))

    # Bias (Systematic Error / Mean Error)
    bias = np.mean(true - pred)

    # SDEP (Standard Deviation of the Error of Prediction / Random Error)
    sdep = np.sqrt(np.mean((true - pred - bias) ** 2))

    return (true, pred, pearson_r2, cod, rmse, bias, sdep)


class Analysis:
    def __init__(
        self,
        rdkit_or_mordred: str = "rdkit",
        held_out_stat_json: str = "held_out_stats.json",
        docking_column: str = "Affinity(kcal/mol)",
    ):
        """
        Description
        -----------
        Class to carry out the Recreating_DMTA workflow analysis

        Parameters
        ----------
        rdkit_or_mordred (str)      Value to set the working results directory, set as 'rdkit' or 'mordred'
        held_out_stats_json (str)   Name of the json file containing the performance stats on the held out
                                    test set
        docking_column (str)        Name of column which the docking scores are saved under

        Returns
        -------
        Initialised analysis class
        """
        global PROJ_DIR

        self.rdkit_or_mordred = rdkit_or_mordred.lower()

        if self.rdkit_or_mordred == "rdkit":
            self.results_dir = PROJ_DIR / "results" / "rdkit_desc"
        else:
            self.results_dir = PROJ_DIR / "results" / "mordred_desc"

        self.held_out_stat_json = held_out_stat_json

        self.docking_column = docking_column

    def _get_stats(
        self,
        experiment_dirs: list = [],
        perf_stats_json: str = "performance_stats.json",
    ):
        """
        Description
        -----------
        Function to get the performance statistics for all experiments on both the internal and held out tests

        Parameters
        ----------
        experiment_dirs (list)      List of experiment names e.g., [20240910_10_r, 20241012_10_r]

        Returns
        -------
        Dictionary containing all available iteration statistics on internal and held out tests for each given expetiment
        """

        all_stats = {}

        # Looping through all provided experiments
        for exp in experiment_dirs:

            if "_50_" in exp:
                step = 50
            else:
                step = 10

            # Initialising empty lists
            rmse = []
            r2 = []
            bias = []
            sdep = []

            no_mols_ls = []

            # Defining the working directory
            working_dir = self.results_dir / exp

            # For each iteration obtain and save the statistics data
            # If this doesnt work change back to (0, cnt_n_iters())
            for n in range(0, count_number_iters(working_dir)):
                no_mols_ls.append(n * step)

                stats_path = f"{working_dir}/it{n}/{perf_stats_json}"

                try:
                    with open(stats_path, "r") as perf_stats:
                        data = json.load(perf_stats)

                    rmse.append(round(float(data.get("RMSE", 0)), 3))
                    r2.append(round(float(data.get("r2", 0)), 3))
                    bias.append(round(float(data.get("Bias", 0)), 3))
                    sdep.append(round(float(data.get("SDEP", 0)), 3))

                except Exception as e:
                    print(e)

            # Format the statistics data
            all_stats[exp] = {
                "n_mols": no_mols_ls,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
                "sdep": sdep,
            }

        return all_stats

    def Plot_Perf(
        self,
        experiments: list,
        save_plot: bool = True,
        results_dir: str = f"{PROJ_DIR}/results/rdkit_desc/",
        plot_fname: str = "Perf_Plot",
        plot_int: bool = False,
        plot_ho: bool = False,
        plot_chembl_int: bool = False,
        r2_ylim: tuple = (-1, 1),
        bias_ylim: tuple = (-0.5, 0.5),
        rmse_ylim: tuple = (0, 1),
        sdep_ylim: tuple = (0, 1),
    ):

        # Load performance stats for the selected datasets
        all_int_stats = (
            self._get_stats(
                experiment_dirs=experiments, perf_stats_json="performance_stats.json"
            )
            if plot_int
            else None
        )
        all_ho_stats = (
            self._get_stats(
                experiment_dirs=experiments,
                perf_stats_json="/held_out_test/held_out_stats.json",
            )
            if plot_ho
            else None
        )
        all_chembl_stats = (
            self._get_stats(
                experiment_dirs=experiments,
                perf_stats_json="chembl_performance_stats.json",
            )
            if plot_chembl_int
            else None
        )

        print(all_ho_stats[experiments[0]]["rmse"][0])

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        linestyles = {"_10_": "-", "_50_": "--"}

        # Determine the maximum length of metrics from any of the datasets to pad missing data
        max_length = 0
        for stats in [all_int_stats, all_ho_stats, all_chembl_stats]:
            if stats:
                max_length = max(
                    max_length, max(len(s["rmse"]) for s in stats.values())
                )

        for exp in experiments:
            step = 50 if "_50_" in exp else 10
            exp_name = exp[9:]
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")
            linestyle = linestyles.get("_50_" if "_50_" in exp else "_10_", "-")

            print(f"Experiment: {exp}, Color: {colour}, Line Style: {linestyle}")

            def plot_metric(ax, stats, metric, linestyle, color, label=None):
                padded_data = np.pad(
                    stats[exp][metric],
                    (0, max_length - len(stats[exp][metric])),
                    constant_values=np.nan,
                )
                sns.lineplot(
                    x=list(range(0, max_length * step, step)),
                    y=padded_data,
                    legend=False,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                    ax=ax,
                )

            if plot_int:
                plot_metric(
                    ax[0, 0],
                    all_int_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_int_stats,
                    "r2",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_int_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_int_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            if plot_ho:
                plot_metric(
                    ax[0, 0],
                    all_ho_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_ho_stats,
                    "r2",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_ho_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_ho_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            if plot_chembl_int:
                plot_metric(
                    ax[0, 0],
                    all_chembl_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_chembl_stats,
                    "r2",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_chembl_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_chembl_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            ax[0, 0].set_title("RMSE")
            ax[0, 0].set_ylabel("RMSE")
            ax[0, 0].set_ylim(rmse_ylim[0], rmse_ylim[1])

            ax[1, 0].set_title("r2")
            ax[1, 0].set_ylabel("r2")
            ax[1, 0].set_ylim(r2_ylim[0], r2_ylim[1])

            ax[1, 1].set_title("Bias")
            ax[1, 1].set_ylabel("Bias")
            ax[1, 1].set_ylim(bias_ylim[0], bias_ylim[1])

            ax[0, 1].set_title("SDEP")
            ax[0, 1].set_ylabel("SDEP")
            ax[0, 1].set_ylim(sdep_ylim[0], sdep_ylim[1])

            for a in ax.flat:
                a.set_xlabel("Molecule Count")

        labels = pd.Series([e[12:] for e in experiments])
        handles = [
            Line2D([0], [0], color=colours, lw=1)
            for colours in method_color_map.values()
        ]

        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        fig.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )

        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_plot:
            plt.savefig(results_dir + "/plots/" + plot_fname + ".png", dpi=600)

        plt.show()

    def PCA_Plot(
        self,
        train: str,
        validation: str,
        prediction: str,
        source_ls: list,
        n_components: int = 5,
        loadings_filename: str = "pca_loadings",
        pca_df_filename: str = "pca_components",
        kdep_sample_size: float = 0.33,
        contamination: float = 0.00001,
        plot_fname: str = "PCA_Plot",
        save_plot: bool = True,
        results_dir: str = f"{PROJ_DIR}/results/rdkit_desc/",
    ):
        """
        Description
        -----------
        Function to do a PCA analysis on the training, validation and prediction molecule sets

        Parameters
        ----------
        train (str)                 File pathway to the features used to train ML models
        validation (str)            File pathway to the features used in a validation/held out test set
        prediction (str)            File pathway to the features used to make predictions
                                    (this requires a general pathway with * to replace batch numbers
                                    e.g., "/path/to/desc/PMG_rdkit_desc_*" )
        source_ls (list)            List of the datasets comparing. (e.g., ChEMBL, Held Out, PyMolGen)
        n_components (int)          Number of principal components to create and plot
        loadings_filename (str)     Name to save the loadings DataFrame under
        pca_df_filename (str)       Name to save the PC DataFrame under
        plot_fname (str)         Name to save the PCA plots under
        kdep_sample_size (float)    Set to decimal for the size of the sample to do the
                                    Kernel Density Estimate Plot from (0.33 = 33 %)
        contamination (float)       Fracion of outlying molecules to remove from the PCA data

        Returns
        -------
        Plot of a n_components x n_components PCA scatter plot.
        """

        # Reading in the training data. Setting the name of the data to colour plot by
        train_df = pd.read_csv(train, index_col="ID")
        used_cols = train_df.columns
        train_df["Source"] = source_ls[0]

        # Reading in the validation data. Setting the name of the data to colour plot by
        validation_df = pd.read_csv(validation, index_col="ID")
        validation_df = validation_df[used_cols]
        validation_df["Source"] = source_ls[1]

        # Reading in the prediction data. Setting the name of the data to colour plot by
        prediction_df = pd.DataFrame()
        files = glob(prediction)
        for file in files:
            df = pd.read_csv(file, index_col="ID")
            df = df[used_cols]
            prediction_df = pd.concat([prediction_df, df], axis=0)
        prediction_df["Source"] = source_ls[2]

        # Making a dictionary for the data and sorting by data length
        # This means that the larger data sets are at the bottom and dont cover smaller ones
        df_dict = {
            "train": train_df,
            "validation": validation_df,
            "prediction": prediction_df,
        }
        sorted_dfs = sorted(df_dict.items(), key=lambda x: len(x[1]), reverse=True)
        combined_df = pd.concat([df for _, df in sorted_dfs], axis=0)

        # Scaling the data between 0 and 1
        scaler = StandardScaler()
        scaled_combined_df = combined_df.copy().dropna()
        scaled_combined_df[used_cols] = scaler.fit_transform(combined_df[used_cols])

        # Doing the PCA on the data
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_combined_df[used_cols])
        explained_variance = pca.explained_variance_ratio_ * 100

        # Isolating the loadings for each principal components and labelling the associated features
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=used_cols
        )
        loadings_df = loadings_df.abs()
        loadings_df.rename_axis("Features", inplace=True)
        loadings_df.to_csv(
            f"{PROJ_DIR}/scripts/run/{loadings_filename}.csv", index_label="Features"
        )

        # Creating a DataFrame for the principal component results. Saves to .csv
        pca_df = pd.DataFrame(
            principal_components,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=combined_df.index,
        )
        pca_df["Source"] = combined_df["Source"].values
        pca_df.to_csv(f"{PROJ_DIR}/scripts/run/{pca_df_filename}.csv", index_label="ID")

        # Removing outlying molecules from the PCA data
        def remove_outliers(df, columns, n_neighbors=20, contamination=contamination):
            """
            Description
            -----------
            Function to remove outlying molecules from a DataFrame

            Parameters
            ----------
            df (pd.DataFrame)       DataFrame from which you wish to remove the outlying molecules
            columns (list)          Columns you want to consider when defining outliers
            contamination (float)   Fraction of outlying points you wish to remove from the dataset

            Returns
            -------
            New DataFrame with outlying molecules removed

            """
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )
            outlier_labels = lof.fit_predict(df[columns])
            return df[outlier_labels == 1]

        pca_df = remove_outliers(pca_df, [f"PC{i+1}" for i in range(n_components)])

        # Initialise PCA subplots
        fig, axs = plt.subplots(
            nrows=n_components, ncols=n_components, figsize=(15, 15)
        )

        # Filling in the subplots
        for i in range(n_components):
            for j in range(n_components):

                # If not on the diagonal, make a scatter plot for the PCA overlap
                if i != j:
               glob     sns.scatterplot(
                        x=f"PC{j+1}",
                        y=f"PC{i+1}",
                        hue="Source",
                        data=pca_df,
                        ax=axs[i, j],
                        legend=False,
                        edgecolor="none",
                        palette="dark",
                    )

                    # Remove x and y plots from every PCA plot apart from the left and bottom most plots
                    if i != n_components - 1:
                        axs[i, j].set_xlabel("")
                        axs[i, j].set_xticks([])

                    if j != 0:
                        axs[i, j].set_ylabel("")
                        axs[i, j].set_yticks([])

                # If on the diagonal, make the Kernel Density Estimate Plots for each Principal Component
                else:
                    # Because this is slow, you can take a sample of the principal component data rather than using the full data
                    src1_data = pca_df[pca_df["Source"] == source_ls[0]]
                    subset_src1_data = src1_data.sample(
                        n=int(len(src1_data) * kdep_sample_size)
                    )

                    src2_data = pca_df[pca_df["Source"] == source_ls[1]]
                    subset_src2_data = src2_data.sample(
                        n=int(len(src2_data) * kdep_sample_size)
                    )

                    src3_data = pca_df[pca_df["Source"] == source_ls[2]]
                    subset_src3_data = src3_data.sample(
                        n=int(len(src3_data) * kdep_sample_size)
                    )

                    sampled_pca_df = pd.concat(
                        [subset_src1_data, subset_src2_data, subset_src3_data], axis=0
                    )

                    # Making the Kernel Density Estimate Plot
                    sns.kdeplot(
                        x=f"PC{i+1}",
                        hue="Source",
                        data=sampled_pca_df,
                        common_norm=False,
                        fill=True,
                        ax=axs[i, i],
                        legend=False,
                        palette="dark",
                    )

                    axs[i, i].set_xlabel("")
                    axs[i, i].set_ylabel("Density Estimate")

                # Adjusting labels and titles, including the variance for each principal component
                if i == n_components - 1:
                    axs[i, j].set_xlabel(
                        f"PC{j+1} ({explained_variance[j]:.2f}% Variance)"
                    )
                if j == 0:
                    axs[i, j].set_ylabel(
                        f"PC{i+1} ({explained_variance[i]:.2f}% Variance)"
                    )

        # Define handles and labels for the legend
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_plot:
            plt.savefig(
                results_dir + "/plots/" + plot_fname + ".png",
                dpi=600,
            )

        return

    def Conformer_Analysis(
        self,
        docking_dir: str = f"{PROJ_DIR}/docking/PyMolGen/",
        sample_size: int = 15,
        conf_gen_plot: bool = True,
        score_convergence: bool = True,
    ):
        """
        Description
        -----------
        Function to look into the conformer generation process. Plots the number of conformers generated for a sample of molecules
        along with the number of docked conformers. Can also plot the highest docking score by iteration to see how far through the
        conformer search are we finding the best docking scores.

        Parameters
        ---------
        docking_dir (str)           Directory to find all of the docking .tar.gz files in
        sample_size (int)           Number of molecules to consider at any given time
        conf_gen_plot (bool)        Flag to plot the number of conformers made (and docked) for each molecule
        score_convergence (bool)    Flag to plot the convergence of highest docking scores by iteration

        Returns
        -------
        None
        """

        # Initiailising empty lists to be used
        n_confs_ls = []
        n_docked_ls = []
        molid_ls = []
        scores_ls = []
        pose_ls = []

        # Obtaining all of the molid files available
        tar_files = glob(docking_dir + "PMG*.tar.gz")

        # Taking a random sample of molecules from all available
        for file in random.sample(tar_files, sample_size):
            file = Path(file)

            # Obtaining just the Molecule ID (.stem removes the .gz suffix)
            molid = file.stem[:-4]

            # Make temporary directory to investigate data in
            output_dir = PROJ_DIR / "docking" / "PyMolGen" / f"extracted_{molid}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # tar command to unzip and untar the molecule docking dataa
            command = ["tar", "-xzf", str(file), "-C", str(output_dir)]
            try:
                subprocess.run(command, check=True)

                # Unzip the .csv.gz file
                try:
                    # Trying to unzip the all_scores file, if fails continues onto next molecule ID
                    gz_file = output_dir / molid / f"{molid}_all_scores.csv.gz"
                    docked_confs_df = pd.read_csv(gz_file)

                    # Updating lists with necessary data
                    scores_ls.append(docked_confs_df[self.docking_column].tolist())
                    n_docked_confs = len(docked_confs_df)
                    n_docked_ls.append(n_docked_confs / 9)
                    pose_ls.append(docked_confs_df.index)

                    # Counting the number of conformations in the all_confs .sdf file
                    n_total_confs = count_conformations(
                        f"{output_dir}/{molid}/all_confs_{molid}_pH74.sdf"
                    )
                    n_confs_ls.append(n_total_confs)
                except:
                    continue

                # Remove the extracted directory
                rm_command = ["rm", "-r", str(output_dir)]
                subprocess.run(rm_command, check=True)

                # Adding the molecule ID to the list if successful
                molid_ls.append(molid)
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract {file}. Error: {e}")

        if conf_gen_plot:

            # Creating a pd.DataFrame with all of the necessary data to make the
            # number of conformers per molecule ID plot

            conf_df = pd.DataFrame(
                {
                    "n_confs_made": n_confs_ls,
                    "molids": molid_ls,
                    "n_confs_docked": n_docked_ls,
                }
            )

            conf_df.index = molid_ls

            # Making the scatter plots
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_made")
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_docked")

            # Formatting the scatter plots
            plt.title("Conformer Generation Analysis")
            plt.xticks(rotation=90)
            plt.ylabel("Number of conformers made")
            plt.xlabel("Molecule ID")
            plt.tight_layout()
            plt.savefig("/users/yhb18174/Recreating_DMTA/scripts/run/conf_gen_plot.png")
            plt.show()

        if score_convergence:

            # Initialising an empty list for all normalised scores
            all_norm_score_lists = []

            for ds_ls in scores_ls:

                # Finding the best scores after each iteration
                best_score = 0
                best_score_ls = []
                for score in ds_ls:
                    if score <= best_score:
                        best_score = score
                    best_score_ls.append(best_score)

                # Normalising the scores between 0 and 1
                min_score = min(best_score_ls)
                max_score = max(best_score_ls)
                if max_score == min_score:
                    normalised_scores = [0.5] * len(best_score_ls)
                else:
                    normalised_scores = [
                        (score - min_score) / (max_score - min_score)
                        for score in best_score_ls
                    ]

                # Updating the normalised scores list
                all_norm_score_lists.append(normalised_scores)

            # Plot the best score lists
            plt.figure()
            for best_score_ls, molid in zip(all_norm_score_lists, molid_ls):
                plt.plot(best_score_ls, label=molid, alpha=0.5)

            # Formatting the plots
            plt.xlabel("Pose Number")
            plt.ylabel("Best Score")
            plt.title("Best Scores Over Time")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")

            plt.tight_layout()
            plt.savefig(
                "/users/yhb18174/Recreating_DMTA/scripts/run/conf_conv_plot.png"
            )
            plt.show()

        return

    def Prediction_Development(
        self,
        experiment: str,
        iter_ls: list,
        n_plots: int=16,
        prediction_fpath: str = "/held_out_test/held_out_test_preds.csv",
        true_path: str = f"{PROJ_DIR}/datasets/held_out_data/PMG_held_out_targ_trimmed.csv",
        dot_size: int = 3,
        save_plot: bool=True,
        plot_filename: str = "preds_dev_plot.png",
        tl_box_position: tuple = (0.35, 0.95),
        br_box_position: tuple = (0.95, 0.05),
        underlay_it0: bool=False,
        regression_line_colour: str = 'gold',
        x_equals_y_line_colour: str = 'red',
        it0_dot_colour: str='purple',
        it_dot_colour: str='teal'
    ):
        """
        Description
        -----------
        Function to look at the true vs predicted values over the iterations for a given test set.
        It will take an even distribution of iteration data from n number of iterations and plot them in a grid.

        Parameters
        ----------
        experiment (str)        Name of the experiment (e.g., 20240910_10_r)
        iter_ls (list)          Iterations to plot recommended for 10 and 50 molecule selections are as follows:
                                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        n_plots (int)           Number of plots to make/iterations to consider. Needs to be a square number (e.g., 4, 9, 16, etc.)
        prediction_fpath (str)  Path to the predicted values for each iteration, considering the pathing is the same.
                                (Only need to specify the dirs after the iteration dir e.g.,
                                DO -> '/held_out_test/preds.csv
                                NOT -> '/path/to/results/it{n}/held_out_test/preds.csv)
        true_path (str)         Pathway to true data .csv file
        dot_size (int)          Size of the dots on the scatter plot
        plot_filename (str)     Name to save the plot under

        Returns
        -------
        None
        """

        step = 50 if "_50_" in experiment else 10

        # Defining the results directory
        working_dir = str(self.results_dir) + "/" + experiment

        # Counting the number of iterations within the working directory
        n_its = count_number_iters(working_dir)

        n_y_plots = int(np.sqrt(n_plots))
        n_x_plots = n_y_plots

        # Reading in the true values
        true_scores = pd.read_csv(true_path, index_col="ID")[self.docking_column]

        # Obtaining the iteration numbers which will be considered.
        # Evenly picks n_plots number of iterations out of the total data
        #its_to_plot = np.round(np.linspace(1, n_its, n_plots)).astype(int).tolist()
    
        its_to_plot = iter_ls

        # Initialising the subplots
        fig, ax = plt.subplots(nrows=n_x_plots, ncols=n_y_plots, figsize=(12, 12))

        it0_preds_df = pd.read_csv(working_dir + "/it0/held_out_test/held_out_test_preds.csv", index_col='ID')
        it0_df = pd.DataFrame()
        it0_df[self.docking_column] = true_scores
        it0_df[f'pred_{self.docking_column}'] = it0_preds_df[f"pred_{self.docking_column}"].tolist()

        # Saving the prediciton dataframes
        df_list = []
        for it in its_to_plot:
            it_dir = working_dir + f"/it{it}"
            preds = it_dir + prediction_fpath

            pred_df = pd.read_csv(preds, index_col="ID")
            pred_df[self.docking_column] = true_scores
            df_list.append(pred_df)

        # Plotting the results
        for i, (df, iter) in enumerate(zip(df_list, its_to_plot)):
            row = i // n_x_plots
            col = i % n_y_plots

            if underlay_it0:
                sns.scatterplot(
                    data=it0_df,
                    x=self.docking_column,
                    y=f'pred_{self.docking_column}',
                    ax=ax[row,col],
                    s=dot_size,
                    color=it0_dot_colour
                )

            sns.scatterplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                s=dot_size,
                color=it_dot_colour
            )

            # Add line of best fit using regplot
            sns.regplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                scatter=False,
                line_kws={"linestyle": "-", "color": regression_line_colour},
            )

            # Calculate the slope of the line of best fit (regression line)
            slope, intercept = np.polyfit(
                df[self.docking_column], df[f"pred_{self.docking_column}"], 1
            )

            # Plot y=x line for reference
            min_val = min(
                df[self.docking_column].min(), df[f"pred_{self.docking_column}"].min()
            )
            max_val = max(
                df[self.docking_column].max(), df[f"pred_{self.docking_column}"].max()
            )
            ax[row, col].plot(
                [min_val, max_val], [min_val, max_val], color=x_equals_y_line_colour, linestyle="-"
            )

            # Calculate angle between the regression line and x=y
            # angle_rad = np.arctan(np.abs((slope - 1) / (1 + slope * 1)))
            # angle_deg = round(np.degrees(angle_rad), 2)

            # Calculate stats
            true, pred, pearson_r2, cod, rmse, bias, sdep = CalcStats(df)
            #print(f"Iteration: {iter}")
            # print(
            #     f"Stats\npr:{pearson_r2}\nr2:{cod}\nrmse:{rmse}\nbias:{bias}\nsdep:{sdep}"
            # )

            avg_pred = np.mean(pred)

            ax[row, col].set_title(f"Iteration {iter} ({iter * step} mols)", fontsize=12)

            # Add text box with metrics
            br_textstr = (
                f"Avg Pred: {avg_pred:.2f}\n"
                f"$R^2_{{cod}}$: {cod:.2f}\n"
                f"$R^2_{{pear}}$: {pearson_r2:.2f}\n"
                # f"Angle: {angle_deg}"
                f"Grad: {round(slope, 2)}"
            )

            tl_textstr = (
                f"RMSE: {rmse:.2f}\n" f"$Bias$: {bias:.2f}\n" f"$sdep$: {sdep:.2f}"
            )

            ax[row, col].text(
                br_box_position[0],
                br_box_position[1],
                br_textstr,
                transform=ax[row, col].transAxes,
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

            ax[row, col].text(
                tl_box_position[0],
                tl_box_position[1],
                tl_textstr,
                transform=ax[row, col].transAxes,
                fontsize=7,
                verticalalignment="top",
                horizontalalignment="right",
            )

            # Set axis labels only for bottom and left most plots
            if row == np.sqrt(n_plots) - 1:  # Bottom row
                ax[row, col].set_xlabel(self.docking_column)
            else:
                ax[row, col].set_xlabel("")

            if col == 0:  # Left-most column
                ax[row, col].set_ylabel(f"pred_{self.docking_column}")
            else:
                ax[row, col].set_ylabel("")
        
        legend_elements = [
            Line2D([0], [0], color='red', linestyle=':', label='x=y'),
            Line2D([0], [0], color='gold', linestyle=':', label='Line of\nBest Fit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='teal', markersize=10, label='Working it\n preds'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=it0_dot_colour, markersize=10, label='it0 preds') if underlay_it0 else None,
        ]

        legend_elements = [elem for elem in legend_elements if elem is not None]

        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.94), ncol=1)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96], pad=1)
        plt.suptitle(f"Prediction Development for {experiment}", fontsize=16, fontweight='bold')

        fig.set_size_inches(12, 10)

        if save_plot:
            plt.savefig(working_dir + "/" + plot_filename, dpi=600)

        plt.show()

        return

    def _pairwise_similarity(self, fngpts_x: list, fngpts_y: list):
        """
        Description
        -----------
        Function to calculate the Tanimoto Similarity matrix between two lists of SMILES strings

        Parameters
        ----------
        fngpts_x (list)     List of molecular fingerprints
        fngpts_y (list)     List of molecular fingerprints

        Returns
        -------
        Similarity matrix for fingerprints x and y
        """

        n_fngpts_x = len(fngpts_x)
        n_fngpts_y = len(fngpts_y)

        similarities = np.zeros((n_fngpts_x, n_fngpts_y))

        for i, fp_x in enumerate(fngpts_x):
            for j, fp_y in enumerate(fngpts_y):
                similarities[i, j] = FingerprintSimilarity(fp_x, fp_y)

        return similarities

    def Tanimoto_Heat_Maps(
        self,
        smiles_a: list,
        smiles_b: list,
        molids_a: list,
        molids_b: list,
        save_plots: bool = False,
        save_path: str = None,
    ):
        """
        Description
        -----------
        Function which takes 2 lists of smiles as inputs and plots the Tanimoto Similarities.
        This analyses both within and across the two lists giving a comprehensive look into the structural similarities

        Parameters
        ----------
        smiles_a (list)         list of SMILES strings
        smiles_b (list)         list of SMILES strings
        molids_a (list)         list of molecule IDs for labelling axes
        molids_b (list)         list of molecule IDs for labelling axes
        save_plots (bool)       flag to save the Tanimoto Similarity heat plots

        Returns
        -------
        Figure containing 3 heat plots:
            1- Tanimoto Similarity between SMILES in smiles_a
            2- Tanimoto Similarity between SMILES in smiles_b
            3- Tanimoto Similarity between SMILES in smiles_a and smiles_b
        """
        mols_a = [Chem.MolFromSmiles(smi) for smi in smiles_a]
        mols_b = [Chem.MolFromSmiles(smi) for smi in smiles_b]

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        fngpts_a = [rdkit_gen.GetFingerprint(mol) for mol in mols_a]

        fngpts_b = [rdkit_gen.GetFingerprint(mol) for mol in mols_b]

        sim_a = self._pairwise_similarity(fngpts_x=fngpts_a, fngpts_y=fngpts_a)

        sim_b = self._pairwise_similarity(fngpts_x=fngpts_b, fngpts_y=fngpts_b)

        sim_ab = self._pairwise_similarity(fngpts_x=fngpts_a, fngpts_y=fngpts_b)

        def heatmap(sim, x_labels, y_labels, ax):
            plot = sns.heatmap(
                sim,
                annot=True,
                annot_kws={"fontsize": 10},
                cmap="crest",
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                cbar=False,
            )

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        heatmap(sim=sim_a, x_labels=molids_a, y_labels=molids_a, ax=axes[0])
        axes[0].set_title("Heatmap Smiles A")

        heatmap(sim=sim_b, x_labels=molids_b, y_labels=molids_b, ax=axes[1])
        axes[1].set_title("Heatmap Smiles B")

        heatmap(sim=sim_ab, x_labels=molids_a, y_labels=molids_b, ax=axes[2])
        axes[2].set_title("Heatmap Smiles A vs Smiles B")

        cbar = fig.colorbar(
            axes[0].collections[0],
            ax=axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        cbar.set_label("Tanimoto Similarity")

        if save_plots:
            plt.savefig(save_path + ".png", dpi=(600))

        plt.show()

    def Avg_Tanimoto_Avg_Across_Iters(
        self,
        experiments: list,
        smiles_df: str = str(PROJ_DIR)
        + "/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv",
        prefix: str = "PMG-",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Tanimoto_Plot",
    ):
        """
        Dictionary
        ----------
        Function to calculate the average pairwise Tanimoto Similarity of the added training molecules
        for each experiment provided and plot them.

        Parameters
        ----------
        experiments (list)          List of experiment names (name of directories results are in)
        smiles_df (str)             Generic pathway to the .csv file containing all of the SMILES
                                    data (uses glob, e.g., /path/to/file/smiles_df_* )
        results_dir (str)           Pathway to results directory where the experiment directories are held
        save_plot (bool)            Flag to save generated plots
        save_path (str)             Pathway to directory you want to save the plots in
        filename (str)              Name of the file to save plots as
        """

        plt.figure(figsize=(10, 6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        tan_sim_dict = {}

        for i, exp in tqdm(
            enumerate(experiments), desc="Processing Experiments", unit="exp"
        ):
            experiment_dir = results_dir + exp
            n_iters = count_number_iters(experiment_dir)

            all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])

            if "__50__" in exp:
                step = 50
            else:
                step = 10

            avg_tanimoto_sim_ls = []
            iter_ls = []
            n_mols_chosen = []

            for iter in range(0, n_iters + 1):
                temp_df = pd.DataFrame()
                start_iter = iter
                end_iter = iter + 1

                molids = get_sel_mols_between_iters(
                    experiment_dir=experiment_dir,
                    start_iter=start_iter,
                    end_iter=end_iter,
                )
                temp_df["ID"] = molids

                smiles = molid_ls_to_smiles(
                    molids=molids, prefix=prefix, data_fpath=smiles_df
                )
                temp_df["SMILES"] = smiles

                mols = [Chem.MolFromSmiles(smi) for smi in smiles]
                temp_df["Mols"] = mols

                added_fngpts = [rdkit_gen.GetFingerprint(mol) for mol in mols]
                temp_df["Fingerprints"] = added_fngpts

                all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
                iter_ls.append(end_iter)
                n_mols_chosen.append(end_iter * step)

                sim = self._pairwise_similarity(
                    fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
                )

                avg_sim = round(np.mean(sim), 4)
                avg_tanimoto_sim_ls.append(avg_sim)
                tan_sim_dict[exp] = avg_tanimoto_sim_ls

            sns.lineplot(
                x=n_mols_chosen, y=avg_tanimoto_sim_ls, label=exp, color=colours[i]
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average Tanimoro Similarity")
        plt.ylim(0, 1)
        plt.title("Average Tanimoto Similarity of Chosen Mols")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600)

        plt.show()

        return tan_sim_dict

    def Top_Preds_Analysis(
        self,
        experiments: list,
        preds_fname: str = "all_preds*",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        preds_column: str = "pred_Affinity(kcal/mol)",
        n_mols: int = 1000,
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Top_Preds_Plot",
        sort_by_descending: bool = True,
    ):

        ascending = False if sort_by_descending else True

        # plt.figure(figsize=(10,6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)
        linestyles = {"_10_": "-", "_50_": "--"}
        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        for exp in tqdm(experiments, desc="Processing Experiments", unit="exp"):
            avg_top_preds = []
            n_mols_chosen = []
            n_iters = count_number_iters(results_dir + exp)

            step = 50 if "_50_" in exp else 10
            linestyle = linestyles["_50_"] if "_50_" in exp else linestyles["_10_"]
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")

            chosen_mols = pd.read_csv(
                results_dir + exp + "/chosen_mol.csv", index_col="ID"
            ).index

            for iter in range(0, n_iters + 1):
                working_dir = Path(results_dir + exp + f"/it{iter}/")
                print(working_dir)
                preds_files = glob(str(working_dir) + "/" + preds_fname)
                top_preds_ls = []

                for file in preds_files:
                    preds = pd.read_csv(file)
                    top_preds = preds.sort_values(
                        by=preds_column, ascending=ascending
                    ).head(n_mols)
                    top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                    top_preds_ls.extend(top_preds[preds_column].tolist())

                top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
                abs_top_preds = top_preds_df.sort_values(
                    by=preds_column, ascending=ascending
                ).head(n_mols)
                avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
                n_mols_chosen.append(iter * step)

            sns.lineplot(
                x=n_mols_chosen,
                y=avg_top_preds,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )

        plt.xlabel("Number of Molecules")
        plt.ylabel(f"Average {preds_column}")
        plt.title(f"Average {preds_column} of top {n_mols} molecules")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600, bbox_inches="tight")

        plt.show()

    def _calc_avg_tanimoto_exp(self, exp, results_dir, smiles_df, prefix):
        experiment_dir = results_dir + exp
        n_iters = count_number_iters(experiment_dir)

        step = 50 if "_50_" in exp else 10

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])
        n_mols_chosen = []
        avg_tanimoto_sim_ls = []

        for iter in range(0, n_iters + 1):
            temp_df = pd.DataFrame()
            start_iter = iter
            end_iter = iter + 1

            molids = get_sel_mols_between_iters(
                experiment_dir=experiment_dir, start_iter=start_iter, end_iter=end_iter
            )

            temp_df["ID"] = molids

            temp_df["SMILES"] = molid_ls_to_smiles(
                molids=molids, prefix=prefix, data_fpath=smiles_df
            )
            temp_df["Mols"] = [Chem.MolFromSmiles(smi) for smi in temp_df["SMILES"]]

            temp_df["Fingerprints"] = [
                rdkit_gen.GetFingerprint(mol) for mol in temp_df["Mols"]
            ]

            all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
            n_mols_chosen.append(end_iter * step)

            sim = self._pairwise_similarity(
                fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
            )

            avg_sim = round(np.mean(sim), 4)
            avg_tanimoto_sim_ls.append(avg_sim)

        return exp, avg_tanimoto_sim_ls, n_mols_chosen, step

    def MP_Avg_Tanimoto_Across_Iters(
        self,
        experiments: list,
        smiles_df: str = str(PROJ_DIR)
        + "/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv",
        prefix: str = "PMG-",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Tanimoto_Plot",
    ):
        """
        Dictionary
        ----------
        Function to calculate the average pairwise Tanimoto Similarity (with Multiprocessing) of the added training molecules
        for each experiment provided and plot them.

        Parameters
        ----------
        experiments (list)          List of experiment names (name of directories results are in)
        smiles_df (str)             Generic pathway to the .csv file containing all of the SMILES
                                    data (uses glob, e.g., /path/to/file/smiles_df_* )
        results_dir (str)           Pathway to results directory where the experiment directories are held
        save_plot (bool)            Flag to save generated plots
        save_path (str)             Pathway to directory you want to save the plots in
        filename (str)              Name of the file to save plots as
        """

        plt.figure(figsize=(10, 6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)

        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        with Pool() as pool:
            results = pool.starmap(
                self._calc_avg_tanimoto_exp,
                [(exp, results_dir, smiles_df, prefix) for exp in experiments],
            )

        for exp, avg_tanimoto_sim_ls, n_mols_chosen, step in results:
            linestyle = "-" if step == 10 else "--"
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")

            sns.lineplot(
                x=n_mols_chosen,
                y=avg_tanimoto_sim_ls,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average Tanimoro Similarity")
        plt.ylim(0, 1)
        plt.title("Average Tanimoto Similarity of Chosen Mols")

        labels = pd.Series([e[12:] for e in experiments])
        handles = [
            Line2D([0], [0], color=colours, lw=1)
            for colours in method_color_map.values()
        ]

        plt.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        plt.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )

        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600)

        plt.show()

        tan_sim_dict = {exp: avg_sim for exp, avg_sim, _, _ in results}

        return tan_sim_dict

    def _process_top_preds_exp(
        self,
        exp: str,
        results_dir: str,
        preds_fname: str,
        preds_column: str,
        n_mols: int,
        ascending: bool,
        linestyles: dict,
        method_colour_map: dict,
    ):

        avg_top_preds = []
        n_mols_chosen = []
        n_iters = count_number_iters(results_dir + exp)

        step = 50 if "_50" in exp else 10
        linestyle = linestyles["_50_"] if "_50_" in exp else linestyles["_10_"]
        method = next((m for m in method_colour_map.keys() if exp.endswith(m)), None)
        colour = method_colour_map.get(method, "black")
        chosen_mols = pd.read_csv(
            results_dir + exp + "/chosen_mol.csv", index_col="ID"
        ).index

        for iter in range(0, n_iters + 1):
            working_dir = Path(results_dir + exp + f"/it{iter}/")
            preds_files = glob(str(working_dir) + "/" + preds_fname)
            top_preds_ls = []

            for file in preds_files:
                preds = pd.read_csv(file)
                top_preds = preds.sort_values(
                    by=preds_column, ascending=ascending
                ).head(n_mols)
                top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                top_preds_ls.extend(top_preds[preds_column].tolist())

            top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
            abs_top_preds = top_preds_df.sort_values(
                by=preds_column, ascending=ascending
            ).head(n_mols)
            avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
            n_mols_chosen.append(iter * step)

        return exp, n_mols_chosen, avg_top_preds, colour, linestyle

    def MP_Top_Preds_Analysis(
        self,
        experiments: list,
        preds_fname: str = "all_preds*",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        preds_column: str = "pred_Affinity(kcal/mol)",
        n_mols: int = 1000,
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Top_Preds_Plot",
        ascending: bool = False,
    ):

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        linestyles = {"_10_": "-", "_50_": "--"}

        method_colour_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_rmp": colours[2],
            "_rmpo": colours[3],
            "_mpo": colours[4],
            "_r": colours[5],
        }

        process_func = partial(
            self._process_top_preds_exp,
            results_dir=results_dir,
            preds_fname=preds_fname,
            preds_column=preds_column,
            n_mols=n_mols,
            ascending=ascending,
            linestyles=linestyles,
            method_colour_map=method_colour_map,
        )

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, experiments),
                    total=len(experiments),
                    desc="Processing Experiments",
                )
            )

        plt.figure(figsize=(10, 6))

        for exp, n_mols_chosen, avg_top_preds, colour, linestyle in results:
            sns.lineplot(
                x=n_mols_chosen,
                y=avg_top_preds,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )


        plt.xlabel("Number of Molecules")
        plt.ylabel(f"Average {preds_column}")
        description = "top" if not ascending else "bottom"
        plt.title(f"Average {preds_column} of {description} {n_mols} mols")

        labels = pd.Series([e[12:] for e in experiments])
        handles = [
            Line2D([0], [0], color=colours, lw=1)
            for colours in method_colour_map.values()
        ]

        plt.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        plt.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600, bbox_inches="tight")
        plt.show()