import logging
import os

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

from data_process import DataProcess
from model import MLModel, Trainer


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
    task = "TSLW"
    select_feature = "f-value"  # pvalue, f-value
    max_features_num = 2048
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
    xgb_params = {
        "learning_rate": [0.1],
        "n_estimators": [150],
        "max_depth": [12],
        "min_child_weight": [2],
        "gamma": [0.4],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "objective": ["reg:squarederror"],
        "reg_alpha": [0.6],
        "reg_lambda": [1.5]
    }

    ml = MLModel(
        model_name="xgboost",
        param_grid=xgb_params
    )
    trainer = Trainer(
        model=ml
    )

    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(X),
        np.asarray(label),
        train_size=0.8,
        random_state=33
    )
    trainer.train(
        x_train=x_train,
        y_train=y_train,
    )
    trainer.eval(x_test=x_test, label=y_test)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    main()
