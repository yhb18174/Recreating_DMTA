import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from joblib import Parallel, delayed
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math


class RF_model:
    def __init__(self):
        """
        Description
        -----------
        Initialising the ML models class
        """
        return

    def _set_inner_cv(self, cv_type: str = "kfold", n_splits: int = 5):
        """
        Description
        -----------
        Setting up the Cross Validation for the inner loop. (Add to this as needed)

        Parameters
        ----------
        cv_type (str)       Name of Cross-Validation type. Current compatible CVs:
                            'kfold'
        n_splits (int)      Number splits to perform

        Returns
        -------
        The object for inner CV.

        """
        rng = rand.randint(0, 2**31)

        if cv_type == "kfold":
            self.inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=rng)

        return self.inner_cv

    def _calculate_performance(
        self, feature_test: pd.DataFrame, target_test: pd.DataFrame, best_rf: object
    ):
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
        true = target_test.astype(float)
        pred = predictions.astype(float)
        errors = true - pred


        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (np.mean((true-pred-(np.mean(true-pred)))**2))**0.5
        mse = mean_squared_error(true, pred)
        rmse = mse**0.5
        r2 = r2_score(true, pred)

        return bias, sdep, mse, rmse, r2, true, pred

    def _plot_feature_importance(
        self,
        feat_importance_df: pd.DataFrame = None,
        top_n_feats: int = 20,
        save_data: bool = False,
        save_path: str = None,
        filename: str = None,
        dpi: int = 500,
    ):
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

        Returns
        -------
        None
        """

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df.head(top_n_feats),
            x="Importance",
            y="Feature",
            palette="viridis",
            dodge=False,
            legend=False,
        )

        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        if save_data:
            plt.savefig(f"{save_path}/{filename}.png", dpi=dpi)

        return

    def _calc_mpo(self,
                  full_data_fpath,
                  preds_df,
                  preds_col_name
                  ):
        df = pd.read_csv(full_data_fpath, index_col='ID', usecols=['ID', 'PFI', 'oe_logp'])
        df[preds_col_name] = preds_df[preds_col_name]
        df['MPO'] = [-score * 1/(1 + math.exp(PFI -8)) for score, PFI in zip(preds_df[preds_col_name], df['PFI'])]
    
        return df

    def Predict(
        self,
        feats: pd.DataFrame,
        save_preds: bool,
        preds_save_path: str = None,
        preds_filename: str = None,
        final_rf: str = None,
        pred_col_name: str = "affinity_pred",
        calc_mpo: bool=True,
        full_data_fpath: str=None
    ):
        """
        Descripton
        ----------
        Function to take make predictions using the input RF model

        Parameters
        ----------
        feats (pd.DataFrame)        DataFrame object containing all of the features used for predictions
        save_preds (bool)           Flag to save the predictions
        preds_save_path (str)       Path to save the predictions to
        preds_filename (str)        Name to save the .csv.gz prediction dfs to
        final_rf (str)              Path to the RF pickle file used to make predictions
        pred_col_name (str)         Name of the column in filename to save predictions to

        Returns
        -------
        pd.DataFrame object containing all of the predictions
        """

        if final_rf is not None:
            rf_model = joblib.load(final_rf)
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame()
        preds_df[pred_col_name] = rf_model.predict(feats)
        preds_df.index = feats.index

        all_tree_preds = np.stack(
            [tree.predict(feats) for tree in rf_model.estimators_]
        )

        if calc_mpo:
            preds_df = self._calc_mpo(full_data_fpath,
                                      preds_df=preds_df,
                                      preds_col_name=pred_col_name
                                      )
            
        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)

        if save_preds:
            preds_df.to_csv(
                f"{preds_save_path}/{preds_filename}.csv.gz",
                index_label='ID',
                compression="gzip",
            )

        return preds_df

    def _fit_model_and_evaluate(
        self,
        n: int,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        test_size: float,
        save_interval_models: bool,
        save_path: str,
        hyper_params: dict
    ):
        """
        Description
        -----------
        Function to carry out single resample and evaluate the performance of the predictions

        Parameters
        ----------
        n (int)                      Resample number
        features (pd.DataFrame)      Training features used to make predictions
        targets (pd.DataFrame)       Training targets used to evaluate training
        test_size (float)            Decimal of test set size (0.3 = 70/30 train/test split)
        save_interval_models (bool)  Flag to save the best rf model from each resample
        save_path (str)              Pathway to save interval models to

        Returns
        -------
        1: best parameters from the hyperparameters search
        2: Performance metrics from the best RF from the given resample
        3: Feature importances from each RF
        """

        rng = rand.randint(0, 2**31)

        print(f"Performing resample {n + 1}")
        resample_number = n + 1

        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features, targets, test_size=test_size, random_state=rng
        )

        # Convert features to DF with feature names
        feat_tr = pd.DataFrame(feat_tr, columns=features.columns)
        feat_te = pd.DataFrame(feat_te, columns=features.columns)

        # Convert DataFrames to NumPy arrays if necessary
        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        # Reinitialize the model and cross validation
        rf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor())
        ])
        
        self._set_inner_cv(cv_type=self.inner_cv_type, n_splits=self.n_splits)

        if self.search_type == "grid":
            search = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                cv=self.inner_cv,
                scoring=self.scoring,
            )
        else:
            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=hyper_params,
                n_iter=self.n_resamples,
                cv=self.inner_cv,
                scoring=self.scoring,
                random_state=rand.randint(0, 2**31),
            )

        search.fit(feat_tr, tar_tr)

        best_pipeline = search.best_estimator_
        best_rf = best_pipeline.named_steps['rf']

        performance = self._calculate_performance(
            target_test=tar_te, feature_test=feat_te, best_rf=best_rf
        )
        
        cross_val_scores = cross_val_score(search.best_estimator_,
                                           feat_tr,
                                           tar_tr,
                                           cv=self.inner_cv,
                                           scoring=self.scoring)

        true_vals_ls = performance[5]
        pred_vals_ls = performance[6]
        performance = performance[:-2]

        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}{n}.pkl")

        return search.best_params_, performance, best_rf.feature_importances_, resample_number, true_vals_ls, pred_vals_ls, np.mean(cross_val_scores)

    def Train_Regressor(
        self,
        search_type: str,
        scoring: str = 'neg_mean_squared_error',
        n_resamples: int = 50,
        inner_cv_type: str = "kfold",
        n_splits: int = 5,
        test_size: float = 0.3,
        hyper_params: dict = None,
        features: pd.DataFrame = None,
        targets: pd.DataFrame = None,
        save_interval_models: bool = False,
        save_path: str = None,
        save_final_model: bool = False,
        plot_feat_importance: bool = False,
        batch_size: int=2
    ):
        """
        Description
        -----------
        Function to train the RF Regressor model.

        Parameters
        ----------
        search_type (str)               Type of hyperparameter search:
                                        'grid' = grid search, exhaustive and more computationally expensive
                                        'random' = random search, non-exhaustive and less computationally expensive
        scoring (str)                   Loss function to map the hyperparameter optimisation to
        n_resamples (int)               Number of Outer Loop resamples
        inner_cv_type (str)             Setting the inner Cross-Validation method
        n_splits (int)                  Number of splits in the inner Cross-Validation
        test_size (float)               Decimal fort he train/test split. 0.3 = 70/30
        hyper_params (dict)             Dictionary of hyperparameters to optimise on
        features (pd.DataFrame)         Features to train the model on
        targets (pd.DataFrame)          Targets to train the model on
        save_interval_models (bool)     Flag to save best individual models from each resample
        save_path (str)                 Path to save individual models to
        save_final_model (bool)         Flag to save the final model after all resampling
        plot_feat_importance (bool)     Flag to plot the feature importance generated by RF model

        Returns
        -------
        1: Final Random Forect model in pickle format
        2: Best hyper parameters for the final model
        3: Dictionary of performance metrics
        4: Dataframe of feature importances
        """
        
        self.inner_cv_type = inner_cv_type
        self.n_splits = n_splits
        self.search_type = search_type
        self.scoring = scoring
        self.n_resamples = n_resamples

        def process_batch(batch_indices):
            results_batch = []
            for n in batch_indices:
                result = self._fit_model_and_evaluate(
                    n, features, targets, test_size, save_interval_models, save_path, hyper_params
                )
                results_batch.append(result)
            return results_batch

        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches = [range(i * batch_size, min((i + 1) * batch_size, n_resamples)) for i in range(n_batches)]
        results_batches = Parallel(n_jobs=-1)(delayed(process_batch)(batch) for batch in batches)
        results = [result for batch in results_batches for result in batch]

        best_params_ls, self.performance_list, feat_importance_ls, self.resample_number_ls, true_vals_ls, pred_vals_ls, self.cross_val_score = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()
        for key, value in best_params.items():
            if key != "rf__max_features":
                best_params[key] = int(value)

        self.performance_dict = {
            "Bias": round(float(np.mean([perf[0] for perf in self.performance_list])), 4),
            "SDEP": round(float(np.mean([perf[1] for perf in self.performance_list])), 4),
            "MSE": round(float(np.mean([perf[2] for perf in self.performance_list])), 4),
            "RMSE": round(float(np.mean([perf[3] for perf in self.performance_list])), 4),
            "r2": round(float(np.mean([perf[4] for perf in self.performance_list])), 4),
        }

        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(), "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)
        if plot_feat_importance:
            print("Plotting feature importance")
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        cleaned_best_params = {key.split('__')[1]: value for key, value in best_params.items()}

        self.final_rf = RandomForestRegressor(**cleaned_best_params)
        self.final_rf.fit(features, targets.to_numpy())

        if save_final_model:
            print(f"Saving final model to:\n{save_path}/final_model.pkl")
            joblib.dump(self.final_rf, f"{save_path}/final_model.pkl")

            with open(f"{save_path}/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file)

            with open(f"{save_path}/best_params.json", "w") as file:
                json.dump(best_params, file)
            
            features.to_csv(f'{save_path}/training_data/training_features.csv.gz', index_label='ID', compression='gzip')
            targets.to_csv(f'{save_path}/training_data/training_targets.csv.gz', index_label='ID', compression='gzip')

        return self.final_rf, best_params, self.performance_dict, feat_importance_df, true_vals_ls, pred_vals_ls, self.cross_val_score

    def AnalyseModel(self):
        mse_list = [perf[2] for perf in self.performance_list]
        
        # Prepare data for seaborn
        data = pd.DataFrame({
            'Resample Number': self.resample_number_ls,
            'MSE': mse_list
        })

        # Create the scatter plot
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x='Resample Number', y='MSE', marker='o', color='blue')
        
        plt.title('Data Leakage Plot')
        plt.xlabel('Resample Number')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.show()

        avg_cv_score = np.mean(self.cross_val_score)
        print(f'Mean cross_val_score: {avg_cv_score}')

        stdev_cv_score = np.std(self.cross_val_score)
        print(f'Standard Deviation cross_val_score: {stdev_cv_score}')

