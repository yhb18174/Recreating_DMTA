import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import joblib
from joblib import Parallel, delayed
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
from tqdm import tqdm
import json
import time


class RF_model():
    def __init__(self):
        
        """
        Description
        -----------
        Initialising the ML models class
        """

        self.inner_cv = None
    
    def _set_inner_cv(self, 
                      cv_type: str='kfold',
                      n_splits: int=5):
        
        rng = rand.randint(0, 2**31)
        
        if cv_type=='kfold':
            self.inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=rng)


        return self.inner_cv

    def _calculate_performance(self,
                               feature_test: pd.DataFrame,
                               target_test: pd.DataFrame,
                               best_rf: object):
        
        """
        Description
        -----------
        Function to calculate the performance metrics used to verify models
        
        Parameters
        ----------
        feature_test (pd.DataFrame)     pd.DataFrame of feature values (x) from the test set
        target_test (pd.DataFrame)       pd.DataFrame of targets (y) from the test set
        best_rf (object)                RF model from the current resample

        Returns
        -------
        Series of performance metrics-
                                        1. Bias
                                        2. Standard Error of Potential
                                        3. Mean Squared Error
                                        4. Root Mean Squared Error (computed from SDEP and Bias)
                                        5. Pearson R coefficient
                                        6. Spearman R coefficient
                                        7. r2 score
        
        """
        # Get predictions from the best model in each resample
        predictions = best_rf.predict(feature_test)

        # Calculate Errors
        errors = target_test - predictions

        # Calculate performance metrics
        # FIX SDEP, MSE

        bias = np.mean(errors)
        sdep = np.std(errors)
        mse = mean_squared_error(target_test, predictions)
        rmsd = np.sqrt(mse)
        r2 = r2_score(target_test, predictions)

        return bias, sdep, mse, rmsd, r2
    
    def _plot_feature_importance(self,
                                 feat_importance_df: pd.DataFrame=None,
                                 top_n_feats: int=20,
                                 save_data: bool=False,
                                 save_path: str=None,
                                 filename: str=None,
                                 dpi: int=500):
        """
        Description
        -----------
        Function to plot feature importance
        
        Parameters
        ----------
        feature_importance_df (pd.DataFrame)    pd.DataFrame containing feature importances
        top_n_feats (int)                       Number of features shown
        save_data (bool)                        Flag to save plot
        save_path (str)                         Path to save plot to
        filename (str)                          Filename to save plot as
        dpi (int)                               Value for quality of saved plot
        """
        
        plt.figure(figsize=(10,8))
        sns.barplot(data=feat_importance_df.head(top_n_feats),
                    x='Importance',
                    y='Feature',
                    hue='Feature',
                    palette='viridis',
                    dodge=False,
                    legend=False)
        
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')

        if save_data:
            plt.savefig(f'{save_path}/{filename}.png', dpi=dpi)

        return
        
    def Train_Regressor(self,
                    search_type: str,
                    scoring: str,
                    n_resamples: int=10,
                    inner_cv_type: str='kfold',
                    n_splits: int=5,
                    test_size: float=0.3,
                    test: bool=False,
                    hyper_params: dict=None,
                    features: pd.DataFrame=None,
                    targets: pd.DataFrame=None,
                    save_interval_models: bool=False,
                    save_path: str=None,
                    save_final_model: bool=False,
                    plot_feat_importance: bool=False):
        """
        Description
        -----------
        Function to train Random Forest Regressor model and calculate the overall performance metrics from all resamples

        Parameters
        ----------
        search_type (str)           The type of search ('grid' or 'random') used for the hyperparameter optimisation process
        scoring (str)               The loss function you want to optimise your model on
        n_resamples (int)           Number of resamples in the outer loop
        inner_cv_type (str)         Set the hyperparameter optimisation CV
        n_splits (int)              Number of splits in the inner CV loop
        test_size (float)           Size of test set. Train/Test split
        test (bool)                 Flag to use a test set. Primarily used to check environments etc
        hyper_params (dict)         Dictionaty of hyperparameters you want to optimise the model on:
                                    {
                                    'n_estimators': [],
                                    'max_features': [],
                                    'max_depth':  [],
                                    'min_samples_split': [],
                                    'min_samples_leaf': []
                                    }
        features (pd.DataFrame)     Input features to train and test the model on, typically referred to as 'X'
        targets (pd.DataFrame)      Input targets to train and test the model on, typically referred to as 'y'
        save_interval_models (bool) Flag to save the best rf_model from each resample
        save_path (str)             Path to save the models to
        save_final_model (bool)     Flag to save final model
        plot_feat_importance (bool) Flag to plot feature importance data

        Returns
        -------
        The final RFR model after optimisation, dictionary of the best parameters found in the optimisation process,
        a dictionary of the average performance metrics, and pd.DataFrame of feature importance
        """

        # Setting up test data (to change with relevant data later) 
        # Hone into hyper parameters       
        if test:
            hyper_params = {'n_estimators': [20, 51],
                            'max_features': ['sqrt'],
                            'max_depth':  [int(10), int(20)],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1]}
            features, targets = make_regression(n_samples=1000, n_features=20, noise=0.5)
            feature_names = [f'Feature {i}' for i in range(features.shape[1])]
            features = pd.DataFrame(features, columns=feature_names)
        else:
            feature_names = features.columns.tolist()

        # Initialise Random Forest Regressor
        self.rf = RandomForestRegressor()
        print('RF Initialised')

        self._set_inner_cv(cv_type=inner_cv_type, n_splits=n_splits)

        # Choose between grid or randomised hyperparameter search
        if search_type == 'grid':
            self.search = GridSearchCV(estimator=self.rf,
                                       param_grid=hyper_params,
                                       cv=self.inner_cv, 
                                       scoring=scoring)
        else:
            self.search = RandomizedSearchCV(estimator=self.rf,
                                             param_distributions=hyper_params,
                                             n_iter=n_resamples,
                                             cv=self.inner_cv,
                                             scoring=scoring,
                                             random_state=rand.randint(0, 2**31))
        print('Hyperparameter search set up')
            
        # Setting up empty performance metric lists to calculate averages from
        best_params_ls = []
        bias_ls = []
        sdep_ls = []
        mse_ls = []
        rmsd_ls = []
        r2_ls = []
        feat_importance_ls = []

        # Start train/test split of data 
        for n in range(n_resamples):
            
            rng = rand.randint(0, 2**31)
            feat_tr, feat_te, tar_tr, tar_te = train_test_split(features, targets, test_size=test_size, random_state=rng)

            print(f'Performing resample {n + 1} of {n_resamples}')

            # Starting hyper parameter optimisation
            self.search.fit(feat_tr, tar_tr)

            # Saving best rf from optimisation
            best_rf = self.search.best_estimator_

            # Calculating performance metrics and adding to their corresponding lists
            performance = self._calculate_performance(target_test=tar_te,
                                                    feature_test=feat_te,
                                                    best_rf=best_rf)

            best_params_ls.append(self.search.best_params_)
            bias_ls.append(performance[0])
            sdep_ls.append(performance[1])
            mse_ls.append(performance[2])
            rmsd_ls.append(performance[3])
            r2_ls.append(performance[4])
            feat_importance_ls.append(best_rf.feature_importances_)

            # Flag to save best rf model from resample cv loop
            if save_interval_models:
                joblib.dump(best_rf, f'{save_path}{n}.pkl')

        # Saving the best hyper parameters from the outer loop resampling
        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()

        for key, value in best_params.items():
            if key == 'max_features':
                continue
            else:
                value = int(value)
            best_params[key] = value
            

        # Calculating overall performance metric averages
        performance_dict = {
            'Bias': round(float(np.mean(bias_ls)), 2),
            'SDEP': round(float(np.mean(sdep_ls)), 2),
            'RMSE': round(float(np.mean(rmsd_ls)), 2),
            'MSE': round(float(np.mean(mse_ls)), 2),
            'r2': round(float(np.mean(r2_ls)), 2)
        }

        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame({'Feature': feature_names,
                                           'Importance': avg_feat_importance})\
                                           .sort_values(by='Importance', ascending=False)
        
        if plot_feat_importance:
            self._plot_feature_importance(save_data=True,
                                          save_path=save_path,
                                          filename='feature_importance_plot')


        # Training new model on the best parameters
        self.final_rf = RandomForestRegressor(**best_params)        
        self.final_rf.fit(features, targets)

        # Saving the best model
        if save_final_model:
            joblib.dump(self.final_rf,  f'{save_path}/final_model.pkl')

        return  self.final_rf, best_params, performance_dict, feat_importance_df
    
    def Predict(self,
                    feats: pd.DataFrame,
                    save_preds: bool,
                    preds_save_path: str=None,
                    preds_filename: str=None,
                    final_rf: str=None,
                    pred_col_name: str='affinity_pred'):
        
        if final_rf is not None:
            with open(final_rf, 'rb') as f:
                rf_model = pk.load(f)
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame()
        preds_df[pred_col_name] = rf_model.predict(feats)
        preds_df.index = feats.index

        all_tree_preds = np.stack([tree.predict(feats) for tree in rf_model.estimators_])
        preds_df['Uncertainty'] = np.std(all_tree_preds, axis=0)

        if save_preds:
            preds_df.to_csv(f'{preds_save_path}/{preds_filename}.csv', index='ID')

        return preds_df
    
    def _fit_model_and_evaluate(self,
                                n,
                                features,
                                targets,
                                test_size,
                                save_interval_models,
                                save_path
                                ):
        
        rng = rand.randint(0, 2**31)

        print(f'Performing resample {n + 1}')


        feat_tr, feat_te, tar_tr, tar_te = train_test_split(features, targets, test_size=test_size, random_state=rng)
        
        # Convert DataFrames to NumPy arrays if necessary
        if isinstance(tar_tr, pd.DataFrame):
            tar_tr = tar_tr.to_numpy().ravel()
        if isinstance(tar_te, pd.DataFrame):
            tar_te = tar_te.to_numpy().ravel()

        self.search.fit(feat_tr, tar_tr)

        best_rf = self.search.best_estimator_

        performance = self._calculate_performance(target_test=tar_te, feature_test=feat_te, best_rf=best_rf)

        if save_interval_models:
            joblib.dump(best_rf, f'{save_path}{n}.pkl')

        return self.search.best_params_, performance, best_rf.feature_importances_
    
    def Train_Regressor_parallel(self,
                                     search_type,
                                     scoring,
                                     n_resamples: int=10,
                                    inner_cv_type: str='kfold',
                                    n_splits: int=5,
                                    test_size: float=0.3,
                                    test: bool=False,
                                    hyper_params: dict=None,
                                    features: pd.DataFrame=None,
                                    targets: pd.DataFrame=None,
                                    save_interval_models: bool=False,
                                    save_path: str=None,
                                    save_final_model: bool=False,
                                    plot_feat_importance: bool=False):
        
        if test:
            hyper_params = {'n_estimators': [20, 51],
                            'max_features': ['sqrt'],
                            'max_depth':  [10, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1]}
            features, targets = make_regression(n_samples=1000, n_features=20, noise=0.5)
            feature_names = [f'Feature {i}' for i in range(features.shape[1])]
            features = pd.DataFrame(features, columns=feature_names)
        else:
            feature_names = features.columns.tolist()

        self.rf = RandomForestRegressor()

        self._set_inner_cv(cv_type=inner_cv_type, n_splits=n_splits)

        if search_type == 'grid':
            self.search = GridSearchCV(estimator=self.rf,
                                   param_grid=hyper_params,
                                   cv=self.inner_cv, 
                                   scoring=scoring)
        else:
            self.search = RandomizedSearchCV(estimator=self.rf,
                                            param_distributions=hyper_params,
                                            n_iter=n_resamples,
                                            cv=self.inner_cv,
                                            scoring=scoring,
                                            random_state=rand.randint(0, 2**31))


        with Parallel(n_jobs=-1) as parallel:
            results = parallel(
                                delayed(self._fit_model_and_evaluate)(
                                n, features, targets, test_size, save_interval_models, save_path
                                ) for n in  tqdm(range(n_resamples), desc="Resamples submitted")
            )

        while len(results) < n_resamples:
            time.sleep(10)

        print('Resamples completed.')

        best_params_ls, performance_ls, feat_importance_ls = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()
        for key, value in best_params.items():
            if key != 'max_features':
                best_params[key] = int(value)

        # Calculating overall performance metric averages
        performance_dict = {
            'Bias': round(float(np.mean([perf[0] for perf in performance_ls])), 2),
            'SDEP': round(float(np.mean([perf[1] for perf in performance_ls])), 2),
            'RMSE': round(float(np.mean([perf[2] for perf in performance_ls])), 2),
            'MSE': round(float(np.mean([perf[3] for perf in performance_ls])), 2),
            'r2': round(float(np.mean([perf[4] for perf in performance_ls])), 2)
                            }       
        
        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame({'Feature': feature_names,
                                        'Importance': avg_feat_importance})\
                                        .sort_values(by='Importance', ascending=False)
        if plot_feat_importance:
            print('Plotting feature importance')
            self._plot_feature_importance(feat_importance_df=feat_importance_df,
                                          save_data=True,
                                          save_path=save_path,
                                          filename='feature_importance_plot')
            
        # Training new model on the best parameters
        self.final_rf = RandomForestRegressor(**best_params)        
        self.final_rf.fit(features, targets.ravel())

        # Saving the best model
        if save_final_model:
            print(f'Saving final model to:\n{save_path}/final_model.pkl')
            joblib.dump(self.final_rf,  f'{save_path}/final_model.pkl')

            with open(f'{save_path}/performance_stats.json', 'w') as file:
                json.dump(performance_dict, file)

            with open(f'{save_path}/best_params.json', 'w') as file:
                json.dump(best_params, file)

        return self.final_rf, best_params, performance_dict, feat_importance_df