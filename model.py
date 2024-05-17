import logging
import random

import numpy as np
from skimage.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from utils import show_fig


class MLModel:
    def __init__(self, model_name, param_grid: dict = None):
        self.param_grid = param_grid or {}
        self.model_name = model_name
        self.cv = 5

        if self.model_name == "xgboost":
            self.model = XGBRegressor()
        elif self.model_name == "linear_model":
            self.model = linear_model.LinearRegression()
        elif self.model_name == "elasticnet":
            self.model = linear_model.ElasticNet()
        else:
            TypeError(f"{self.model_name} not support!")

    def fit(self, x_train, y_train, x_val=None, y_val=None, *args, **kwargs):
        self.model = self.gridsearchcv(self.model, self.param_grid, X=x_train, Y=y_train, *args, **kwargs)
        if self.model_name == "xgboost":
            if x_val is not None and y_val is not None:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val), (x_train, y_train)],
                    eval_metric=["rmse"],
                    early_stopping_rounds=10,
                    verbose=False
                )
                results = self.model.evals_result()
                y_dict = {
                    "Val": results['validation_0']['rmse'],
                    "Train": results['validation_1']['rmse']
                }
                show_fig(y_dict, y_label="RMSE", title="XGBOOST")

    def predict(self, x_test):
        return self.model.predict(x_test)

    def gridsearchcv(self, model, param_grid, X, Y):
        logging.info(f"param_grid: \n{param_grid}")
        grid_search = GridSearchCV(model, param_grid, cv=self.cv)
        grid_search.fit(X, Y)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        best_model = grid_search.best_estimator_
        best_model.fit(X, Y)
        return best_model

    def xgboost(self, model, x_train, y_train, x_val=None, y_val=None) -> BaseEstimator:
        logging.info(f"xgboost model train...\n{self.param_grid}")

        model = XGBRegressor()
        best_model = self.gridsearchcv(
            model=model,
            param_grid=self.param_grid,
            X=x_train,
            Y=y_train,
            cv=5
        )
        if x_val is not None and y_val is not None:
            best_model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val), (x_train, y_train)],
                eval_metric=["rmse"],
                early_stopping_rounds=10,
                verbose=False
            )
            results = best_model.evals_result()
            y_dict = {
                "Val": results['validation_0']['rmse'],
                "Train": results['validation_1']['rmse']
            }
            show_fig(y_dict, y_label="RMSE", title="XGBOOST")

        return best_model

    def linear_model(self, x_train, y_train, x_val=None, y_val=None) -> BaseEstimator:
        logging.info("linear model train...")
        model = linear_model.LinearRegression()

        param_grid = {
            "normalize": [True, False]
        }
        best_model = self.gridsearchcv(model, param_grid, x_train, y_train, 5)
        best_model.fit(x_train, y_train)

        return best_model

    def elasticnet(self, x_train, y_train, x_val=None, y_val=None) -> BaseEstimator:
        logging.info("elasticnet model train...")
        model = linear_model.ElasticNet()
        param_grid = {
            "alpha": [1, 2]
        }
        best_model = self.gridsearchcv(model, param_grid, x_train, y_train, 5)
        best_model.fit(x_train, y_train)
        return best_model


class Trainer:
    def __init__(self, model: MLModel, test_ration=0.8, seed=42):
        self.test_ration = test_ration
        self.seed = seed

        self.set_seed()
        self.model: MLModel = model

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None):
        logging.info(f"train shape: {x_train.shape}, label shape: {y_train.shape}")

        self.model.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    def eval(self, x_test, label):
        assert len(x_test) == len(label), f"predict nums: {len(x_test)} != label nums: {len(label)}"

        predict = self.model.predict(x_test)
        r2 = r2_score(label, predict)
        mse = mean_squared_error(label, predict)
        rmse = np.sqrt(mean_squared_error(label, predict))
        mae = mean_absolute_error(label, predict)
        logging.info(f"r2: {r2}\nmse:{mse}\nrmse:{rmse}\nmae:{mae}")
