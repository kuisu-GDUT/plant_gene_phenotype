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

    @staticmethod
    def gridsearchcv(model, param_grid, X, Y, cv: int = 5):
        grid_search = GridSearchCV(model, param_grid, cv=cv)
        grid_search.fit(X, Y)
        print("Best Parameters: ", grid_search.best_params_)
        best_model = grid_search.best_estimator_
        best_model.fit(X, Y)
        return best_model

    def xgboost(self, x_train, y_train, x_val=None, y_val=None) -> BaseEstimator:
        logging.info("xgboost model train...")
        param_grid = {
            # "learning_rate": [0.1],
            # "n_estimators": [500],
            # "max_depth": [9, 10,11],
            # "min_child_weight": [2],
            # "gamma": [0.4],
            # "colsample_bytree": [0.7],
            # "objective": ["reg:squarederror"],
            # "reg_alpha": [0.55, 0.6, 0.65],
            # "reg_lambda": [0.5, 1, 1.5]
        }

        best_model = XGBRegressor(
            learning_rate=0.01,
            n_estimators=600,  # 树的个数--100棵树建立xgboost
            max_depth=12,  # 树的深度
            min_child_weight=2,  # 叶子节点最小权重
            gamma=0.4,  # 惩罚项中叶子结点个数前的参数
            subsample=0.7,  # 随机选择70%样本建立决策树
            colsample_bytree=0.7,  # 随机选择70%特征建立决策树
            objective='reg:squarederror',  # 使用平方误差作为损失函数
            reg_alpha=2,
            reg_lambda=2,
        )
        # best_model = self.gridsearchcv(
        #     model=model,
        #     param_grid=param_grid,
        #     X=x_train,
        #     Y=y_train,
        #     cv=5
        # )
        # if x_val is not None and y_val is not None:
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
        self.ml = model

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def train(self, df_X, df_Y, df_X_test, df_Y_test, fill_nan=False):
        logging.info(f"train shape: {df_X.shape}, label shape: {df_Y.shape}")
        # if fill_nan:
        #     df_X = df_X.fillna(-1)
        # x_train, x_test, y_train, y_test = train_test_split(
        #     np.asarray(df_X),
        #     np.asarray(df_Y),
        #     train_size=self.test_ration,
        #     random_state=33
        # )

        x_train, x_test, y_train, y_test = [np.array(i) for i in [df_X, df_X_test, df_Y, df_Y_test]]

        model = self.ml.xgboost(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
        predict = model.predict(x_test)
        self.eval(predict, label=y_test)

    def eval(self, predict, label):
        assert len(predict) == len(label), f"predict nums: {len(predict)} != label nums: {len(label)}"

        r2 = r2_score(label, predict)
        mse = mean_squared_error(label, predict)
        rmse = np.sqrt(mean_squared_error(label, predict))
        mae = mean_absolute_error(label, predict)
        logging.info(f"r2: {r2}\nmse:{mse}\nrmse:{rmse}\nmae:{mae}")
