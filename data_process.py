import logging
import os

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression


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

    @staticmethod
    def cal_corr(data: pd.DataFrame, save_dir: str = None):
        corr = data.corr()
        sns.heatmap(corr)
        f, ax = plt.subplots(figsize=(14, 10))
        ax.set_title("Correlation between features")
        plt.show()
        if save_dir is not None:
            if os.path.exists(save_dir):
                f.savefig(os.path.join(save_dir, "corr_heatmap.png"), dpi=100)


def select_features(df_feature_train, label_train, select_feature, task, save_path, max_features_num):
    dp = DataProcess()
    if select_feature == "p-value":
        X_snp_names = [name for name in df_feature_train.columns.values if "snp" in name]
        X_otu_names = [name for name in df_feature_train.columns.values if "OTU" in name]
        X_snp_pvalue = dp.f_regression(data=df_feature_train[X_snp_names], label=label_train)
        X_otu_pvalue1 = dp.linear_regression(data=df_feature_train[X_otu_names[:500]], label=label_train)
        X_otu_pvalue2 = dp.linear_regression(data=df_feature_train[X_otu_names[500:]], label=label_train)
        pvalue_dict = {
            "name": X_snp_names + X_otu_names,
            "pvalue": X_snp_pvalue + X_otu_pvalue1 + X_otu_pvalue2
        }
        pvalue_df = pd.DataFrame(pvalue_dict)
        r_pvalue = pvalue_df.sort_values(by=["pvalue"])
        r_pvalue = r_pvalue[r_pvalue["pvalue"] < 0.05]
        r_pvalue.sort_values(by=["pvalue"], inplace=True)
        r_pvalue.to_csv(os.path.join(save_path, "pvalue/pvalue_order_{}.csv".format(task)))
        select_feature_names = list(r_pvalue["name"].values)[:max_features_num]
    else:
        sel = SelectKBest(score_func=f_regression, k=min(max_features_num, df_feature_train.shape[-1]))
        sel = sel.fit(df_feature_train.fillna(-1), label_train)
        select_feature_names = df_feature_train.columns[sel.get_support(True)]
        fvalue_df = pd.DataFrame(
            {"fvalue": sel.pvalues_, "score": sel.scores_, "name": df_feature_train.columns.values}
        )
        fvalue_df = fvalue_df[fvalue_df["fvalue"] < 0.05]
        fvalue_df.sort_values(by=["fvalue"], inplace=True)
        fvalue_df.to_csv(os.path.join(save_path, "pvalue/fvalue_order_{}.csv".format(task)))
    return select_feature_names
