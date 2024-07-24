import sys
from pathlib import Path

FILE_DIR = Path(__file__).parent
sys.path.insert(0, str(FILE_DIR.parent))

from RF_class import RF_model
import pandas as pd

model=RF_model()
n_resamples = 50
inner_cv_type = 'kfold',
n_splits = 5
tr_te_split = 0.3
search_type = 'grid'
loss_function = 'neg_mean_squared_error'

hyper_params = \
    {
    'n_estimators': [300, 400, 500],
    'max_features': [1/3],
    'max_depth': [50, 100, 150, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':[5]
    }


rdkit_desc_path = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/ChEMBL_rdkit_desc_1.csv.gz'
mordred_desc_path = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/mordred/ChEMBL_mordred_desc_1.csv.gz'
targets_path = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/ChEMBL_docking_df.csv'
rdkit_save_path = '/users/yhb18174/Recreating_DMTA/RF_model/results/rdkit_desc/init_RF_model/it0'
mordred_save_path = '/users/yhb18174/Recreating_DMTA/RF_model/results/mordred_desc/init_RF_model/it0'

rdkit_features = pd.read_csv(rdkit_desc_path, index_col='ID', compression='gzip')
targets_df = pd.read_csv(targets_path, index_col='ID')
targets = targets_df['affinity_exp'].values.ravel()

rf, params, perf, feats = model.Train_Regressor_parallel(search_type=search_type,
                                                   scoring=loss_function,
                                                   n_resamples=n_resamples,
                                                   inner_cv_type=inner_cv_type,
                                                   n_splits=n_splits,
                                                   test_size=tr_te_split,
                                                   test=False,
                                                   hyper_params=hyper_params,
                                                   features=rdkit_features,
                                                   targets=targets,
                                                   save_path=rdkit_save_path,
                                                   save_final_model=True,
                                                   plot_feat_importance=True)


mordred_features = pd.read_csv(mordred_desc_path, index_col='ID')

rf, params, perf, feats = model.Train_Regressor_parallel(search_type=search_type,
                                                   scoring=loss_function,
                                                   n_resamples=n_resamples,
                                                   inner_cv_type=inner_cv_type,
                                                   n_splits=n_splits,
                                                   test_size=tr_te_split,
                                                   test=False,
                                                   hyper_params=hyper_params,
                                                   features=mordred_features,
                                                   targets=targets,
                                                   save_path=mordred_save_path,
                                                   save_final_model=True,
                                                   plot_feat_importance=True)