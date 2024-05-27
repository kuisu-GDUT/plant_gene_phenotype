import argparse
import logging
import os

import pandas as pd

from data_process import DataProcess, select_features
from model import MLModel, Trainer


def create_parameters():
    parser = argparse.ArgumentParser(description='Corn phenotype predict based on the ML')
    parser.add_argument(
        "--snp_data_path",
        type=str,
        default=r"D:\03_data\11_MAT_DATA\06_data\群体数据\玉米\GSTP003\population\gwas\1458_Inbred_gwas.DTT.vcf.raw.head",
        help="snp data path, which is csv file"
    )
    parser.add_argument(
        "--phenotype_data_path",
        type=str,
        default=r"D:\03_data\11_MAT_DATA\06_data\群体数据\玉米\GSTP003\population\GSTP003.pheno",
        help="phenotype data path, "
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\03_data\11_MAT_DATA\06_data\群体数据\玉米\GSTP003\population\results",
        help="output directory"
    )
    parser.add_argument(
        "--task_phenotype",
        type=str,
        default="DTT",
        help="predict phenotype task"
    )
    parser.add_argument(
        "--train_val_test_split_dir",
        type=str,
        default=r"D:\03_data\11_MAT_DATA\06_data\群体数据\玉米\GSTP003\population\gwas",
        help="The data was split into train, val and test with a ratio of 6:2:2"
    )
    return parser.parse_args()


class CornDataProcess(DataProcess):
    @staticmethod
    def read_data(path: str, header=None, sep=None):
        assert os.path.exists(path), f"{path} is not exits"
        logging.info(f"read data path: {path}")

        if os.path.basename(path).endswith("csv"):
            df = pd.read_csv(path, header=header, sep=sep, low_memory=False)
        elif os.path.basename(path).endswith("xlsx"):
            df = pd.read_excel(path, header=header)
        elif len(os.path.basename(path).split(".")) == 1:
            df = pd.read_table(path, header=header, sep="\t")
            names = ["snp_{}".format(name) for name in df["ID"]]
            df = df.iloc[:, 8:].T
            df.columns = names
        else:
            if sep is not None:
                df = pd.read_csv(path, header=header, sep=sep, low_memory=False)
            else:
                raise TypeError("{} not support".format(path))
        logging.info(f"data shape: {df.shape}")
        logging.info(df.head())
        return df

    def merge_data(
            self,
            snp_path: str,
            label_path: str,
            save_path: str,
            one_hot=False,
            normalize=False,
            vt_threshold=0.5,
            otu_path: str = None,
    ):
        df_snp = self.read_snp(snp_path, sep=" ", vt_threshold=None)
        df_labels = self.read_data(label_path, sep="\t", header=0)
        df_labels = df_labels.set_index(df_labels.columns.values[0])
        df_merge = pd.merge(df_labels, df_snp, how="inner", left_index=True, right_index=True)
        logging.info(f"df merge shape: {df_merge.shape}")
        logging.info(f"df merge head:\n{df_merge.head()}")
        if os.path.exists(save_path):
            df_merge.to_csv(os.path.join(save_path, "Y_Xs_merge.csv"))
        return df_merge

    def read_snp(self, snp_path: str, sep=None, vt_threshold=0.5, one_hot=False, header=0):
        df_snp = self.read_data(snp_path, sep=sep, header=header)
        names = [name for name in df_snp.columns.values if name.startswith("chr")]
        assert "IID" in df_snp.columns.values, f"IID must be in the {snp_path}"
        df_snp = df_snp.set_index("IID")
        df_snp = df_snp[names]
        df_snp.columns = ["snp_{}".format(name) for name in df_snp.columns.values]

        if isinstance(vt_threshold, float):
            logging.info(f"vt process...threshold: {vt_threshold}")
            select_idx = self.vt_feature_selection(df_snp, threshold=vt_threshold)
            df_snp = df_snp.iloc[:, select_idx]

        if one_hot:
            logging.info(f"one hot process...")
            df_snp = self.one_hot(df_snp, tokenizer={0: "a", 1: "b", 2: "c"}, nan_token="d")
        logging.info(f"snp shape: {df_snp.shape}")
        return df_snp


def train(x_train, y_train, x_test, y_test):
    xgb_params = {
        "learning_rate": [0.1],
        "n_estimators": [150],
        "max_depth": [12],
        "min_child_weight": [2],
        "gamma": [0.4],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "objective": ["reg:squarederror"],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "reg_lambda": [0.8, 1, 1.5]
    }
    ml = MLModel(
        model_name="xgboost",
        param_grid=xgb_params
    )
    trainer = Trainer(
        model=ml
    )
    train_result = trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test
    )
    eval_result = trainer.eval(x_test=x_test, label=y_test)


def main():
    args = create_parameters()
    logging.info(f"args: {args}")

    cdp = CornDataProcess()
    df = cdp.merge_data(args.snp_data_path, label_path=args.phenotype_data_path, save_path=args.output_dir)
    train_names = cdp.read_data(os.path.join(args.train_val_test_split_dir, "train.pheno"), header=0, sep=" ")[
        "IID"].values
    test_names = cdp.read_data(os.path.join(args.train_val_test_split_dir, "test.pheno"), header=0, sep=" ")[
        "IID"].values

    df_train = df.loc[[name for name in df.index if name in train_names]]
    df_test = df.loc[[name for name in df.index if name in train_names]]
    train_labels = df_train[args.task_phenotype]
    test_labels = df_test[args.task_phenotype]
    train_data = df_train.iloc[:, 3:10]
    test_data = df_train.iloc[:, 3:10]

    train(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
