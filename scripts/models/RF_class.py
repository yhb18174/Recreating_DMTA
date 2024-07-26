import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
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
        errors = target_test - predictions

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = np.std(errors)
        mse = mean_squared_error(target_test, predictions)
        rmsd = np.sqrt(mse)
        r2 = r2_score(target_test, predictions)

        return bias, sdep, mse, rmsd, r2

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
            hue="Feature",
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

    def Predict(
        self,
        feats: pd.DataFrame,
        save_preds: bool,
        preds_save_path: str = None,
        preds_filename: str = None,
        final_rf: str = None,
        pred_col_name: str = "affinity_pred",
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
            with open(final_rf, "rb") as f:
                rf_model = pk.load(f)
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame()
        preds_df[pred_col_name] = rf_model.predict(feats)
        preds_df.index = feats.index

        all_tree_preds = np.stack(
            [tree.predict(feats) for tree in rf_model.estimators_]
        )
        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)

        if save_preds:
            preds_df.to_csv(
                f"{preds_save_path}/{preds_filename}.csv.gz",
                index="ID",
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

        search = self.search
        rng = rand.randint(0, 2**31)

        print(f"Performing resample {n + 1}")

        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features, targets, test_size=test_size, random_state=rng
        )

        # Convert DataFrames to NumPy arrays if necessary
        if isinstance(tar_tr, pd.DataFrame):
            tar_tr = tar_tr.to_numpy().ravel()
        if isinstance(tar_te, pd.DataFrame):
            tar_te = tar_te.to_numpy().ravel()

        search.fit(feat_tr, tar_tr)

        best_rf = search.best_estimator_

        performance = self._calculate_performance(
            target_test=tar_te, feature_test=feat_te, best_rf=best_rf
        )

        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}{n}.pkl")

        return search.best_params_, performance, best_rf.feature_importances_

    def Train_Regressor(
        self,
        search_type: str,
        scoring: str,
        n_resamples: int = 10,
        inner_cv_type: str = "kfold",
        n_splits: int = 5,
        test_size: float = 0.3,
        test: bool = False,
        hyper_params: dict = None,
        features: pd.DataFrame = None,
        targets: pd.DataFrame = None,
        save_interval_models: bool = False,
        save_path: str = None,
        save_final_model: bool = False,
        plot_feat_importance: bool = False,
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
        test (bool)                     Flag to test the function on a much smaller sample of data
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

        if test:
            hyper_params = {
                "n_estimators": [20, 51],
                "max_features": ["sqrt"],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1],
            }
            features, targets = make_regression(
                n_samples=1000, n_features=20, noise=0.5
            )
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]
            features = pd.DataFrame(features, columns=feature_names)
        else:
            feature_names = features.columns.tolist()

        self.rf = RandomForestRegressor()

        self._set_inner_cv(cv_type=inner_cv_type, n_splits=n_splits)

        if search_type == "grid":
            self.search = GridSearchCV(
                estimator=self.rf,
                param_grid=hyper_params,
                cv=self.inner_cv,
                scoring=scoring,
            )
        else:
            self.search = RandomizedSearchCV(
                estimator=self.rf,
                param_distributions=hyper_params,
                n_iter=n_resamples,
                cv=self.inner_cv,
                scoring=scoring,
                random_state=rand.randint(0, 2**31),
            )

        results = []
        # Sequentially process each task without multiprocessing
        for n in tqdm(range(n_resamples), desc="Resamples submitted"):
            result = self._fit_model_and_evaluate(
                n, features, targets, test_size, save_interval_models, save_path
            )
            results.append(result)

        # Attempt of Parallelisation
        # with Parallel(n_jobs=-1, batch_size=5) as parallel:
        #     results = parallel(
        #                         delayed(self._fit_model_and_evaluate)(
        #                         n, features, targets, test_size, save_interval_models, save_path
        #                         ) for n in  tqdm(range(n_resamples), desc="Resamples submitted")
        #     )

        print("Resamples completed.")

        best_params_ls, performance_ls, feat_importance_ls = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()
        for key, value in best_params.items():
            if key != "max_features":
                best_params[key] = int(value)

        # Calculating overall performance metric averages
        performance_dict = {
            "Bias": round(float(np.mean([perf[0] for perf in performance_ls])), 2),
            "SDEP": round(float(np.mean([perf[1] for perf in performance_ls])), 2),
            "RMSE": round(float(np.mean([perf[2] for perf in performance_ls])), 2),
            "MSE": round(float(np.mean([perf[3] for perf in performance_ls])), 2),
            "r2": round(float(np.mean([perf[4] for perf in performance_ls])), 2),
        }

        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)
        if plot_feat_importance:
            print("Plotting feature importance")
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        # Training new model on the best parameters
        self.final_rf = RandomForestRegressor(**best_params)
        self.final_rf.fit(features, targets.ravel())

        # Saving the best model
        if save_final_model:
            print(f"Saving final model to:\n{save_path}/final_model.pkl")
            joblib.dump(self.final_rf, f"{save_path}/final_model.pkl")

            with open(f"{save_path}/performance_stats.json", "w") as file:
                json.dump(performance_dict, file)

            with open(f"{save_path}/best_params.json", "w") as file:
                json.dump(best_params, file)

        return self.final_rf, best_params, performance_dict, feat_importance_df
