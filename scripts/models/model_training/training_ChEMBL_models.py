import sys
from pathlib import Path
from glob import glob
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

FILE_DIR = Path(__file__).parent
sys.path.insert(0, str(FILE_DIR.parent))

from RF_class import RF_model
import pandas as pd

n_resamples = 50
inner_cv_type = ("kfold",)
n_splits = 5
tr_te_split = 0.3
search_type = "grid"
loss_function = "neg_mean_squared_error"
docking_column = "Affinity(kcal/mol)"

hyper_params = {
    "rf__n_estimators": [400, 500],
    "rf__max_features": ["sqrt"],
    "rf__max_depth": [10, 20, 30, 50],
    "rf__min_samples_split": [10, 20, 30],
    "rf__min_samples_leaf": [10, 20, 30],
}

training_path = "/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/"
targs_df = pd.read_csv(training_path + "dock/new_ChEMBL_docking_df.csv", index_col="ID")

# Setting up input targets, removing any ones which failed to dock
targs = targs_df[["Affinity(kcal/mol)"]]
falsetargs = targs_df[targs_df["Affinity(kcal/mol)"] == "False"]
targs = targs.drop(index=falsetargs.index)

feats_df = pd.read_csv(
    training_path + "desc/rdkit/ChEMBL_rdkit_desc_1.csv.gz",
    index_col="ID",
    compression="gzip",
)
save_path = (
    "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/init_RF_model/it0_diff_hyp_2/"
)

new_feat_df = feats_df
new_feat_df = new_feat_df.drop(index=falsetargs.index)

model = RF_model(docking_column=docking_column)

model.Train_Regressor(
    search_type=search_type,
    hyper_params=hyper_params,
    features=new_feat_df,
    targets=pd.DataFrame(targs),
    save_path=save_path,
    save_final_model=True,
    plot_feat_importance=True,
)

# Define file paths and prefixes
desc_fpath = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/"
desc_fprefix = "PMG_rdkit_desc_*.csv"
desc_files = glob(desc_fpath + desc_fprefix)

full_data_fpath = (
    "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/"
)
full_data_fprefix = "PMG_rdkit_*.csv"

for n in range(len(desc_files)):
    print(f"\nPredicting scores for batch {n+1}:")
    desc_pref = desc_fprefix.replace("*", str(n + 1))
    desc_file = desc_fpath + desc_pref

    full_pref = full_data_fprefix.replace("*", str(n + 1))
    full_file = full_data_fpath + full_pref

    feats = pd.read_csv(desc_file, index_col="ID")[new_feat_df.columns]
    preds = model.Predict(
        feats=pd.DataFrame(feats),
        save_preds=True,
        calc_mpo=True,
        full_data_fpath=full_file,
        preds_filename=f"all_preds_{n+1}",
        preds_save_path=save_path,
    )
    print(f"Preds {n+1} complete.")


# Predicting on held out
held_out_dir = "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/"

it_held_out_dir = Path(save_path) / "held_out_test"
it_held_out_dir.mkdir(exist_ok=True)

rf_pkl = f"{save_path}final_model.pkl"
rf_model = joblib.load(rf_pkl)

with open(rf_pkl, "rb") as feats:
    data = pickle.load(feats)

feats_df = pd.read_csv(f"{held_out_dir}/PMG_held_out_desc.csv", index_col="ID")
ho_df = pd.read_csv(f"{held_out_dir}/PMG_held_out_docked.csv", index_col="ID").drop(
    columns=["Unnamed: 0"]
)
ho = ho_df[[docking_column]]
falseho = ho_df[ho_df[docking_column] == "False"]
ho_ = ho.drop(index=falseho.index)
ho_

same_value_columns = feats_df.columns[~feats_df.apply(lambda col: col.nunique() == 100)]
new_feat_df = feats_df[same_value_columns]
new_feat_df = new_feat_df.drop(index=falseho.index)

new_feat_df = new_feat_df[data.tolist()]

preds = rf_model.predict(new_feat_df)
preds_df = pd.DataFrame(index=new_feat_df.index)
preds_df["pred_Affinity(kcal/mol)"] = preds
preds_df.to_csv(str(it_held_out_dir) + "/held_out_preds.csv", index_label="ID")

ho_[f"pred_{docking_column}"] = preds

true = ho_[docking_column].astype(float)
pred = ho_[f"pred_{docking_column}"].astype(float)

# Create scatter plot
sns.scatterplot(data=ho_, x=f"pred_{docking_column}", y=docking_column)

# Get current axis
ax = plt.gca()

# Set ticks for x and y axes
x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
y_ticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.set_title("Predicted DS score vs Actual DS score")

# Show plot
plt.savefig(f"{held_out_dir}/init_preds_vs_actual.png")

errors = true - pred

# Calculate performance metrics
bias = np.mean(errors)
sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
mse = mean_squared_error(true, pred)
rmse = np.sqrt(mse)
r2 = r2_score(true, pred)

dict = {
    "Bias": round(bias, 3),
    "SDEP": round(sdep, 3),
    "MSE": round(mse, 3),
    "RMSE": round(rmse, 3),
    "r2": round(r2, 3),
}

import json

with open(f"{save_path}/held_out_test/held_out_stats.json", "w") as file:
    json.dump(dict, file, indent=4)
