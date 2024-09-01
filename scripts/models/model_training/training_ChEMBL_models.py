import sys
from pathlib import Path

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
    "rf__max_depth": [25, 50, 100],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [2, 4, 8],
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
save_path = "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/init_RF_model/it0"

same_value_columns = feats_df.columns[~feats_df.apply(lambda col: col.nunique() == 1)]
new_feat_df = feats_df[same_value_columns]
new_feat_df = new_feat_df.drop(index=falsetargs.index)

model = RF_model(docking_column=docking_column)

model.Train_Regressor(
    search_type=search_type,
    hyper_params=hyper_params,
    features=new_feat_df,
    targets=targs,
    save_path=save_path,
    save_final_model=True,
    plot_feat_importance=True,
)
