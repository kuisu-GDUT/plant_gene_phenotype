import logging
import os
import random

import numpy as np
import pandas as pd

from matplotlib import pyplot
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics, linear_model
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm


def show_fig(y_dict: dict, y_label: str = None, title: str = None):
    fig, ax = pyplot.subplots()
    for label, y in y_dict.items():
        epochs = len(y)
        x_axis = range(0, epochs)
        ax.plot(x_axis, y, label=label)
        ax.legend()
    pyplot.ylabel(y_label)
    pyplot.title(title)
    pyplot.show()


class DataProcess:

    def merge_data(
            self,
            snp_path: str,
            otu_path: str,
            label_path: str,
            save_path: str,
            one_hot=False,
            normalize=False,
            vt_threshold=0.5
    ):
        """merge snp, otu and label to one file"""
        df_snp = self.read_snp(snp_path, None, one_hot)
        df_snp.to_csv(os.path.join(save_path, "snp.csv"))

        df_otu = self.read_otu(otu_path, vt_threshold, normalize)
        df_otu.to_csv(os.path.join(save_path, "otu.csv"))

        df_label = self.read_label(label_path, header=0)
        df_label.to_csv(os.path.join(save_path, "label.csv"))

        x_data = pd.merge(df_snp, df_otu, how='left', left_index=True, right_index=True)
        Y_X = pd.merge(df_label, x_data, how="left", left_index=True, right_index=True)

        Y_X.to_csv(os.path.join(save_path, "merge.csv"))
        return Y_X

    def read_label(self, label_path: str, header=0):
        df_label = pd.read_csv(label_path, header=header)
        df_label_names = [i for i in df_label["Plant_traits"]]
        df_label = df_label.iloc[:, 1:]
        df_label = df_label.T
        df_label.columns = df_label_names
        logging.info(f"label shape: {df_label.shape}")

        return df_label

    def read_otu(self, otu_path, vt_threshold=0.5, normalize=False, header=0):
        df_otu = pd.read_csv(otu_path, header=header)
        df_otu = df_otu.set_index("ID", drop=True)

        if vt_threshold:
            logging.info(f"vt process...threshold: {vt_threshold}")
            select_idx = self.vt_feature_selection(df_otu, threshold=vt_threshold)
            df_otu = df_otu.iloc[:, select_idx]

        if normalize:
            logging.info("normalize process of otu...")
            df_otu = self.normalization(df_otu)
        logging.info(f"otu shape: {df_otu.shape}")
        return df_otu

    def read_snp(self, snp_path: str, vt_threshold=0.5, one_hot=False, header=0):
        df_snp = self.read_data(snp_path, header=header)
        df_snp_names = ["snp_{}".format(i) for i in df_snp["ID"]]
        df_snp = df_snp.iloc[:, 9:]
        df_snp = df_snp.T
        df_snp.columns = df_snp_names

        if vt_threshold:
            logging.info(f"vt process...threshold: {vt_threshold}")
            select_idx = self.vt_feature_selection(df_snp, threshold=vt_threshold)
            df_snp = df_snp.iloc[:, select_idx]

        if one_hot:
            logging.info(f"one hot process...")
            df_snp = self.one_hot(df_snp, tokenizer={0: "a", 1: "b", 2: "c"}, nan_token="d")
        logging.info(f"snp shape: {df_snp.shape}")
        return df_snp

    @staticmethod
    def one_hot(data: pd.DataFrame, tokenizer: dict, nan_token="d"):
        for key, value in tokenizer:
            data = data.replace(key, value)
        data = data.fillna(nan_token)
        data = pd.get_dummies(data)
        return data

    @staticmethod
    def normalization(data: pd.DataFrame):
        return (data - data.min()) / (data.max() - data.min())

    @staticmethod
    def vt_feature_selection(data: pd.DataFrame, threshold=0.5):
        """删除方差阈值 < 0.5"""
        logging.info(f"thread:{threshold}, before shape: {data.shape}")
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(data)
        select_idx = sel.get_support(True)
        logging.info(f"after select features: {len(select_idx)}")
        logging.info('Variances is %s' % sel.variances_)
        return select_idx

    @staticmethod
    def f_regression(data: pd.DataFrame, label):
        data_fill = data.fillna(-1)
        sel = SelectKBest(score_func=f_regression, k=min(10, data.shape[-1]))
        sel.fit(data_fill, label)
        pvalues = list(sel.pvalues_)
        return pvalues

    @staticmethod
    def linear_regression(data: pd.DataFrame, label):
        data_fill = data.fillna(-1)
        data_fill = sm.add_constant(data_fill)
        lr_model = sm.OLS(label, data_fill).fit()
        return list(lr_model.pvalues.iloc[1:])

    @staticmethod
    def read_data(path: str, header=None):
        assert os.path.exists(path), f"{path} is not exits"
        logging.info(f"read data path: {path}")

        if os.path.basename(path).endswith("csv"):
            df = pd.read_csv(path, header=header, low_memory=False)
        elif os.path.basename(path).endswith("xlsx"):
            df = pd.read_excel(path, header=header)
        elif len(os.path.basename(path).split(".")) == 1:
            df = pd.read_table(path, header=header, sep="\t")
            names = ["snp_{}".format(name) for name in df["ID"]]
            df = df.iloc[:, 8:].T
            df.columns = names
        else:
            raise TypeError("{} not support".format(path))
        logging.info(f"data shape: {df.shape}")
        logging.info(df.head())
        return df


class ML_Model:

    @staticmethod
    def gridsearchcv(model, param_grid, X, Y, cv: int = 5):
        grid_search = GridSearchCV(model, param_grid, cv=cv)
        grid_search.fit(X, Y)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
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
            # "reg_alpha": [0.1, 1, 2],
            # "reg_lambda": [1, 2, 2.5]
        }

        best_model = XGBRegressor(
            learning_rate=0.05,
            n_estimators=300,  # 树的个数--100棵树建立xgboost
            max_depth=12,  # 树的深度
            min_child_weight=2,  # 叶子节点最小权重
            gamma=0.4,  # 惩罚项中叶子结点个数前的参数
            subsample=0.7,  # 随机选择70%样本建立决策树
            colsample_bytree=0.7,  # 随机选择70%特征建立决策树
            objective='reg:squarederror',  # 使用平方误差作为损失函数
            reg_alpha=2,
            reg_lambda=2,
        )
        best_model = self.gridsearchcv(
            model=best_model,
            param_grid=param_grid,
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
    def __init__(self):
        self.train_test_ration = 0.8
        self.seed = 42

        self.set_seed()
        self.ml = ML_Model()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def train(self, df_X, df_Y, fill_nan=False):
        logging.info(f"train shape: {df_X.shape}, label shape: {df_Y.shape}")
        # if fill_nan:
        #     df_X = df_X.fillna(-1)
        x_train, x_test, y_train, y_test = train_test_split(
            np.asarray(df_X),
            np.asarray(df_Y),
            train_size=self.train_test_ration,
            random_state=33
        )

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


def main():
    path_mange = {
        "snp_path": {
            "test_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\all_genotype data_test.csv",
            "actual_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\all_genotype data.csv",
            "one_snp_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827TSLW_SNP4"
        },
        "otu_path": {
            "test_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\微生物数据\核心OTU_827OTU0.7.csv",
            "actual_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\微生物数据\核心OTU_827OTU0.7.csv"
        },
        "label_path": {
            "test_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\谷子表型数据\Millet_12_phenotrypes.rename.csv",
            "actual_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\谷子表型数据\Millet_12_phenotrypes.rename.csv"
        },
        "save_path": {
            "test_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\data_process\test",
            "actual_path": r"D:\03_data\11_MAT_DATA\06_data\谷子\data_process"
        },
        "task_path": {
            "TSLW": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827TSLW_SNP4",
            "MSW": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827MSW_SNP4",
            "MSPD": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827MSPD_SNP4",
            "MSPW": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827MSPW_SNP4",
            "PGW": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827PGW_SNP4",
            "MSPL": r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827MSPL_SNP4",
        }
    }

    snp_path = path_mange["snp_path"]["actual_path"]
    assert os.path.exists(snp_path), f"{snp_path} is not exits."
    otu_path = path_mange["otu_path"]["actual_path"]
    assert os.path.exists(otu_path), f"{otu_path} is not exits."
    label_path = path_mange["label_path"]["actual_path"]
    assert os.path.exists(label_path), f"{label_path} is not exits."
    save_path = path_mange["save_path"]["actual_path"]
    assert os.path.exists(save_path), f"{save_path} is not exits."
    task = "PGW"
    select_feature = "f-value"  # pvalue, f-value
    # max_features_num = 2048
    for max_features_num in [512]:

        logging.info(f"task:{task}\nselect_feature:{select_feature}\nmax_features_num:{max_features_num}")

        dp = DataProcess()
        df_merge = dp.read_data(os.path.join(save_path, "merge.csv"), header=0)
        df_merge = df_merge.set_index("Unnamed: 0", drop=True)

        # merge
        task_snp_path = path_mange["task_path"][task]
        if os.path.exists(task_snp_path):
            task_snp_df = dp.read_data(task_snp_path, header=0)
            new_names = [i for i in task_snp_df.columns.values if i not in df_merge.columns.values]
            df_feature = pd.merge(df_merge, task_snp_df[new_names], how="left", left_index=True, right_index=True)
            logging.info(f"merge shape: {df_feature.shape}, merge snp data: {task_snp_path}")

        df_merge.sample(frac=1).reset_index(drop=True)
        df_feature = df_merge.iloc[:, 12:]
        label = df_merge[task]

        if select_feature == "p-value":
            X_snp_names = [name for name in df_feature.columns.values if "snp" in name]
            X_otu_names = [name for name in df_feature.columns.values if "OTU" in name]
            X_snp_pvalue = dp.f_regression(data=df_feature[X_snp_names], label=label)
            X_otu_pvalue1 = dp.linear_regression(data=df_feature[X_otu_names[:500]], label=label)
            X_otu_pvalue2 = dp.linear_regression(data=df_feature[X_otu_names[500:]], label=label)
            pvalue_dict = {
                "name": X_snp_names + X_otu_names,
                "pvalue": X_snp_pvalue + X_otu_pvalue1 + X_otu_pvalue2
            }
            pvalue_df = pd.DataFrame(pvalue_dict)
            r_pvalue = pvalue_df.sort_values(by=["pvalue"])
            r_pvalue = r_pvalue[r_pvalue["pvalue"] < 0.05]
            pvalue_df.sort_values(by=["pvalue"], inplace=True)
            pvalue_df.to_csv(os.path.join(save_path, "pvalue_order_{}.csv".format(task)))
            select_feature_names = list(r_pvalue["name"].values)[:max_features_num]
        else:
            sel = SelectKBest(score_func=f_regression, k=min(max_features_num, df_feature.shape[-1]))
            sel = sel.fit(df_feature.fillna(-1), label)
            select_feature_names = df_feature.columns[sel.get_support(True)]

        logging.info(f"select feature nums: {len(select_feature_names)}")

        X = df_feature[select_feature_names]
        trainer = Trainer()
        trainer.train(df_X=X, df_Y=label)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    main()
