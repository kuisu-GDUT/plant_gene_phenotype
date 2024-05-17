import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

from data_process import DataProcess, select_features
from model import Trainer, MLModel
from utils import show_fig


class MSPDModel(MLModel):


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
            reg_lambda=2.5,
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
    task = "MSPD"
    select_feature = "f-value"  # pvalue, f-value
    # max_features_num = 2048
    seed = 42
    test_ration = 0.2
    random.seed(seed)
    np.random.seed(seed)
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
        df_merge_test = df_merge.sample(frac=test_ration, replace=True, random_state=seed)
        df_merge_train = df_merge[~df_merge.index.isin(df_merge_test.index)]
        df_feature_train = df_merge_train.iloc[:, 12:]
        label_train = df_merge_train[task]
        label_test = df_merge_test[task]

        select_feature_names = select_features(
            df_feature_train,
            label_train,
            select_feature,
            task,
            save_path,
            max_features_num
        )
        logging.info(f"select feature nums: {len(select_feature_names)}")

        X_train = df_feature_train[select_feature_names]
        X_test = df_merge_test[select_feature_names]

        trainer = Trainer(
            model=MSPDModel()
        )
        trainer.train(df_X=X_train, df_Y=label_train, df_X_test=X_test, df_Y_test=label_test)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    main()
