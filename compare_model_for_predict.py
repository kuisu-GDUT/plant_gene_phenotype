import logging
import os
import random

import numpy as np
import pandas as pd

from data_process import DataProcess, select_features
from model import Trainer, MLModel


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

    tasks = ["TSLW", "MSW", "MSPD", "MSPW", "PGW", "MSPL"]
    select_feature = "f-value"  # pvalue, f-value
    max_features_num = 1024
    fill_nan = True
    max_features_nums = [256, 512, 1024, 2048, 4096]
    seeds = [1, 2, 3, 4, 42]
    test_ration = 0.2
    summary_result = {
        "select_feature": select_feature,
        # "seed": seed,
        "test_ration": test_ration
    }
    df_result = pd.DataFrame()

    ############## Strat ##################################################################
    for task in tasks:
        for max_features_num in max_features_nums:
            for seed in seeds:
                summary_result["task"] = task
                logging.info(f"task:{task}\nselect_feature:{select_feature}\nmax_features_num:{max_features_num}")
                random.seed(seed)
                np.random.seed(seed)
                dp = DataProcess()
                df_merge = dp.read_data(os.path.join(save_path, "merge.csv"), header=0)
                df_merge = df_merge.set_index("Unnamed: 0", drop=True)

                # merge
                task_snp_path = path_mange["task_path"][task]
                if os.path.exists(task_snp_path):
                    task_snp_df = dp.read_data(task_snp_path, header=0)
                    new_names = [i for i in task_snp_df.columns.values if i not in df_merge.columns.values]
                    df_feature = pd.merge(df_merge, task_snp_df[new_names], how="left", left_index=True,
                                          right_index=True)
                    logging.info(f"merge shape: {df_feature.shape}, merge snp data: {task_snp_path}")

                # df_merge.sample(frac=1).reset_index(drop=True)
                # split test and train data
                if fill_nan:
                    df_merge = df_merge.fillna(-1)
                df_merge_test = df_merge.sample(frac=test_ration, replace=True, random_state=seed)
                df_merge_train = df_merge[~df_merge.index.isin(df_merge_test.index)]
                df_feature_train = df_merge_train.iloc[:, 12:]
                label_train = df_merge_train[task]
                label_test = df_merge_test[task]

                select_feature_names = select_features(
                    df_merge.iloc[:, 12:],
                    df_merge[task],
                    select_feature,
                    task,
                    save_path,
                    max_features_num
                )
                logging.info(f"select feature nums: {len(select_feature_names)}")

                X_train = df_feature_train[select_feature_names]
                X_test = df_merge_test[select_feature_names]

                ml = MLModel(
                    model_name="elasticnet",
                    param_grid={
                        'alpha': [0.001, 0.01, 0.1],
                        'l1_ratio': [0.1, 0.5, 0.9]
                    },
                    cv=5
                )
                trainer = Trainer(
                    model=ml
                )
                train_result = trainer.train(
                    x_train=X_train,
                    y_train=label_train,
                    x_val=X_test,
                    y_val=label_test
                )
                eval_result = trainer.eval(x_test=X_test, label=label_test)
                summary_result.update(train_result)
                summary_result.update(eval_result)
                summary_result.update({"features_num": len(select_feature_names)})
                df_result = df_result.append(pd.DataFrame(summary_result, index=[0]))
        df_result.to_csv(os.path.join(save_path, "elastic/all_search_params_{}_{}.csv".format("elastic", task)))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    main()
