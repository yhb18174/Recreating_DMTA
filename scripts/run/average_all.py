import pandas as pd
from pathlib import Path
import numpy as np
import sys
from glob import glob
import json


PROJ_DIR = Path(__file__).parent.parent.parent
FILE_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR) + "/scripts/misc/")

from misc_functions import (
    count_number_iters,
    get_chembl_molid_smi,
    get_sel_mols_between_iters,
    molid_ls_to_smiles,
)

class AverageAll():

    def __init__(
            self,
            results_dir: str=f"{str(PROJ_DIR)}/results/rdkit_desc/",
            docking_column: str = "Affinity(kcal/mol)",
            preds_column: str = "pred_Affinity(kcal/mol)"
    ):
        
        self.results_dir = results_dir
        self.docking_column = docking_column
        self.preds_column = preds_column
    
    def _avg_stats(self,
                   it:int,
                   all_exp_dirs: list):
        
        ho_stats = pd.DataFrame()
        int_stats = pd.DataFrame()
        chembl_int_stats = pd.DataFrame()

        for dir in all_exp_dirs:
            working_dir = str(dir) + f'/it{it}'

            # Load internal performance json
            try:
                with open(working_dir + '/performance_stats.json', 'r') as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                int_stats = pd.concat([int_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)
                
            # Load hold out performance json
            try:
                with open(working_dir + '/held_out_test/held_out_stats.json', 'r') as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                ho_stats = pd.concat([ho_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)

            # Load ChEMBL internal performance json
            try:
                with open(working_dir + '/chembl_performance_stats.json', 'r') as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                chembl_int_stats = pd.concat([chembl_int_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)

        print(f"Internal Statistics:\n{int_stats}\n")
        print(f"Hold Out Statistics:\n{ho_stats}\n")
        print(f"ChEMBL Internal Statistics:\n{chembl_int_stats}\n")
        
        # Convert all data to a dictionary
        avg_int_dict = int_stats.mean().to_dict()
        avg_ho_dict = ho_stats.mean().to_dict()
        avg_chembl_int_dict = chembl_int_stats.mean().to_dict()
        
        return avg_int_dict, avg_ho_dict, avg_chembl_int_dict

    def _avg_feat_importance(self,
                             it: int,
                             all_exp_dirs: list):
        
        avg_feat_df = pd.DataFrame()

        for dir in all_exp_dirs:
            working_dir = str(dir) + f'/it{it}'
            print(working_dir)

            if avg_feat_df.empty:
                try:
                    avg_feat_df = pd.read_csv(working_dir + '/feature_importance_df.csv').sort_index(ascending=True)

                except Exception as e:
                    print(e)
            else:
                loaded_df = pd.read_csv(working_dir + '/feature_importance_df.csv').sort_index(ascending=True)
                merged_df = pd.merge(avg_feat_df, loaded_df, left_index=True, right_index=True, suffixes=("_df1", "_df2"))
                merged_df["Importance"] = merged_df[[f"Importance_df1", f"Importance_df2"]].mean(axis=1)
                merged_df['Feature'] = merged_df["Feature_df1"]

                avg_feat_df = pd.DataFrame({
                                        "Importance": merged_df['Importance'].tolist(),
                                        "Feature": merged_df['Feature'].tolist()
                                    })
                
                avg_feat_df.sort_values(by="Feature", inplace=True)

        return avg_feat_df

    def _average_experiment(
            self,
            exp_suffix: str,
            n_iters: int,
            dir_prefix: str="average_"

    ):
        
        all_dirs = [d for d in Path(self.results_dir).iterdir() if d.is_dir()]
        all_exp_dirs = [exp for exp in all_dirs if str(exp).endswith(exp_suffix) and not str(exp.name).startswith("20241011") and not str(exp.name).startswith("average")]

        print(f"Taking averages over experiments:\n")
        for dir in all_exp_dirs: 
            print(dir.name)
            
        # make average dir
        dir_path = Path(f"{self.results_dir}/{dir_prefix}{exp_suffix}")

        dir_path.mkdir(parents=True, exist_ok=True)
                
        for it in range (0, n_iters + 1):
            # make it dir, ho_dir
            working_dir = f"{str(dir_path)}/it{it}/"
            working_path = Path(working_dir)
            working_path.mkdir(parents=True, exist_ok=True)

            working_ho = Path(f"{str(dir_path)}/it{it}/held_out_test/")
            working_ho.mkdir(parents=True, exist_ok=True)

            print(f"==================  Iteration: {it}  ==================")
            # Averaging Statistics

            avg_int, avg_ho, avg_chembl = self._avg_stats(it=it,
                                                          all_exp_dirs=all_exp_dirs)
            
            # Save to corresponding locations
            with open(f"{working_dir}/performance_stats.json", "w") as f:
                json.dump(avg_int, f, indent=4)      

            with open(f"{working_ho}/held_out_stats.json", "w") as f:
                json.dump(avg_ho, f, indent=4)    

            with open(f"{working_dir}/chembl_performance_stats.json", "w") as f:
                json.dump(avg_chembl, f, indent=4)       

            # Averaging Feature Importance
            avg_feat_df = self._avg_feat_importance(it=it,
                                                    all_exp_dirs=all_exp_dirs)
            
            # Save to corresponding locations
            avg_feat_df.to_csv(f"{working_dir}/feature_importance_df.csv")