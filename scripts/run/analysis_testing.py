#%%
import sys
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, PROJ_DIR / 'scripts/run/')
from analysis_class import Analysis

sys.path.insert(0, PROJ_DIR / 'scripts/misc/')
from misc_functions import get_sel_mols_between_iters, molid_to_smiles, molid_ls_to_smiles


an = Analysis()

#%%
an.Plot_Perf(experiments=['20240910_10_mu',
                        '20240910_10_mp', 
                        '20240910_10_r',
                        '20240910_10_rmp',
                        '20240910_10_rmpo',
                        '20240910_10_mpo',
                        '20240916_50_mu',
                        '20240916_50_mp',
                        '20240916_50_r',
                        '20240916_50_rmp', 
                        '20240916_50_rmpo',
                        '20240916_50_mpo'])

#%%

# chembl_feats = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/ChEMBL_rdkit_desc_1.csv.gz'
# ho_feats = '/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_desc.csv'
# prediction = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/PMG_rdkit_desc*'

# an.PCA_Plot(train=chembl_feats,
#             validation=ho_feats,
#             prediction=prediction,
#             source_ls=['ChEMBL', 'Held_Out', 'PyMolGen'],
#             plot_filename='3rd_pca_plot_test')

# %%
# an.Prediction_Development("20240910_10_mu") 
# %%
# molids_a = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/20240910_10_mp',
#                            start_iter = 1,
#                            end_iter= 1)
# molids_b = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/20240910_10_mp',
#                            start_iter = 2,
#                            end_iter= 2)

# smiles_a = molid_ls_to_smiles(molids_a, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")
# smiles_b = molid_ls_to_smiles(molids_b, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")

# %%
# an.Tanimoto_Heat_Maps(smiles_a, smiles_b, molids_a, molids_b)

# %%
# an.Avg_Tanimoto_Avg_Across_Iters(experiments=['20240910_10_mu',
#                                               '20240910_10_mp', 
#                                               '20240910_10_r',
#                                               '20240910_10_rmp',
#                                               '20240910_10_rmpo',
#                                               '20240910_10_mpo',
#                                               '20240916_50_mu',
#                                               '20240916_50_mp',
#                                               '20240916_50_r',
#                                               '20240916_50_rmp', 
#                                               '20240916_50_rmpo',
#                                               '20240916_50_mpo'])

# %%
# an._get_top_preds(experiments=['20240910_10_mu',
#                                 '20240910_10_mp', 
#                                 '20240910_10_r',
#                                 '20240910_10_rmp',
#                                 '20240910_10_rmpo',
#                                 '20240910_10_mpo',
#                                 '20240916_50_mu',
#                                 '20240916_50_mp',
#                                 '20240916_50_r',
#                                 '20240916_50_rmp', 
#                                 '20240916_50_rmpo',
#                                 '20240916_50_mpo'])
# %%
