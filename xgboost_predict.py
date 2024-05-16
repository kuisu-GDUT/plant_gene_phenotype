import logging
import os
import random

import numpy as np
import pandas as pd

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics, linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
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
        sel = SelectKBest(score_func=f_regression, k=1024)
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
        else:
            raise TypeError("{} not support".format(path))
        logging.info(f"data shape: {df.shape}")
        logging.info(df.head())
        return df


class ML_Model:
    @staticmethod
    def xgboost(x_train, y_train, x_test, y_test) -> BaseEstimator:
        model = XGBRegressor(
            learning_rate=0.1,
            n_estimators=300,  # 树的个数--100棵树建立xgboost
            max_depth=12,  # 树的深度
            min_child_weight=2,  # 叶子节点最小权重
            gamma=0.4,  # 惩罚项中叶子结点个数前的参数
            subsample=0.7,  # 随机选择70%样本建立决策树
            colsample_bytree=0.7,  # 随机选择70%特征建立决策树
            objective='reg:squarederror',  # 使用平方误差作为损失函数
            random_state=1,  # 随机数
            # reg_alpha=2,
            # reg_lambda=2,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test), (x_train, y_train)],
            # early_stopping_rounds=10,
            eval_metric=["rmse"],
            verbose=True
        )
        results = model.evals_result()
        y_dict = {
            "Test": results['validation_0']['rmse'],
            "Train": results['validation_1']['rmse']
        }
        show_fig(y_dict, y_label="RMSE", title="XGBOOST")

        return model

    def linear_model(self, x_train, y_train, x_test=None, y_test=None) -> BaseEstimator:
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        return model

    def elasticnet(self, x_train, y_train, x_test=None, y_test=None) -> BaseEstimator:
        model = linear_model.ElasticNet()
        model.fit(x_train, y_train)
        return model


class Trainer:
    def __init__(self):
        self.train_test_ration = 0.8
        self.seed = 42

        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def train(self, df_X, df_Y):
        x_train, x_test, y_train, y_test = train_test_split(
            np.asarray(df_X),
            np.asarray(df_Y),
            train_size=self.train_test_ration,
            random_state=33
        )

        model = ML_Model.xgboost(x_train, y_train, x_test, y_test)
        predict = model.predict(x_test)
        self.eval(predict, label=y_test)

    def eval(self, predict, label):
        assert len(predict) == len(label), f"predict nums: {len(predict)} != label nums: {len(label)}"

        r2 = r2_score(label, predict)
        mse = mean_squared_error(label, predict)
        rmse = np.sqrt(mean_squared_error(label, predict))
        mae = mean_absolute_error(label, predict)
        logging.info(f"r2: {r2}\nmse:{mse}\nrmse:{rmse}\nmae:{mae}")


def readData(path):
    '''读取数据'''
    data = pd.read_excel(path)
    print('path:{}, 读取的表格的形状:{}'.format(path, data.shape))
    if "Unnamed: 0" in data.columns.values:
        data.drop("Unnamed: 0", axis=1, inplace=True)
        print("deleted feature of 'Unnamed: 0'")
    return data


def Evaluate(model, X, Y, predict):
    '''
    预测结果评估
    :param y:
    :param predict:
    :return:
    '''
    ### 模型正确率
    accuracy = accuracy_score(Y, predict)
    print("准确率: %.2f%%" % (accuracy * 100.0))

    confM = confusion_matrix(Y, predict)
    print('confusion_matrix:\n{}'.format(confM))
    disp = metrics.plot_confusion_matrix(model, X, Y)
    plt.show()
    print(f"Classification report for classifier {model}:\n"
          f"{metrics.classification_report(Y, predict)}\n")
    # 回归评价指标(MSE)
    from sklearn.metrics import mean_squared_error  # MSE
    mse = mean_squared_error(predict, Y)
    print('mse:{}'.format(mse))


def model_xgb(x_train, y_train, x_test, y_test, mode="xgboost"):
    # fig, ax = plt.subplots(figsize=(15, 15))#设置图像大小
    # xgboost
    ### 训练模型
    if mode == "xgboost":
        model = XGBRegressor(
            learning_rate=0.1,
            n_estimators=300,  # 树的个数--100棵树建立xgboost
            max_depth=12,  # 树的深度
            min_child_weight=2,  # 叶子节点最小权重
            gamma=0.4,  # 惩罚项中叶子结点个数前的参数
            subsample=0.7,  # 随机选择70%样本建立决策树
            colsample_bytree=0.7,  # 随机选择70%特征建立决策树
            objective='reg:squarederror',  # 使用平方误差作为损失函数
            random_state=1,  # 随机数
            # reg_alpha=2,
            # reg_lambda=2,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test), (x_train, y_train)],
            # early_stopping_rounds=10,
            eval_metric=["rmse"],
            verbose=True
        )
        predictions = model.predict(x_test)
    elif mode == "xgb":
        params = {'learning_rate': 0.1,
                  'max_depth': 10,  # 构建树的深度，越大越容易过拟合
                  'num_boost_round': 2000,
                  'objective': 'reg:squarederror',  # 线性回归问题
                  # 'objective': 'reg:linear',      # 线性回归问题，早期版本的参与，将被reg:squarederror替换
                  'random_state': 7,
                  'gamma': 0,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'reg_alpha': 0.005,
                  'n_estimators': 1000,
                  'eval_metric': ['logloss', 'rmse', 'mae'],  # 分类有“auc”
                  'eta': 0.3  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
                  }
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        res = xgb.cv(params, dtrain, num_boost_round=5000, metrics='rmse', early_stopping_rounds=25)
        best_nround = res.shape[0] - 1
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        evals_result = {}
        model = xgb.train(params, dtrain, evals=watchlist, evals_result=evals_result)
        predictions = model.predict(xgb.DMatrix(x_test))
    elif mode == "LR":
        from sklearn import linear_model

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
    elif mode == "EN":
        from sklearn import linear_model

        model = linear_model.ElasticNet()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    # predictions = model.predict(x_test)
    # predictions = [round(value) for value in predictions]
    print('r2_score:', r2_score(y_test, predictions))
    print('mse:', mean_squared_error(predictions, y_test))
    print('rmse:', np.sqrt(mean_squared_error(predictions, y_test)))
    print('mae:', mean_absolute_error(predictions, y_test))
    # print('r2:', r2_score(predictions, y_test))
    # evaluate predictions
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Test')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Train')
    ax.legend()
    pyplot.ylabel('rmse')
    pyplot.title('XGBoost MSE')
    pyplot.show()
    # plot classification error
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['error'], label='Train')
    # ax.plot(x_axis, results['validation_1']['error'], label='Test')
    # ax.legend()
    # pyplot.ylabel('Classification Error')
    # pyplot.title('XGBoost Classification Error')
    # pyplot.show()

    return model


def feature_selection_top_200(data: pd.DataFrame, y_label: pd.DataFrame, k: int = None):
    snp_feature_idx = []
    otu_feature_idx = []
    for name in data.columns.values:
        if "snp" in name:
            snp_feature_idx.append(data.columns.get_loc(name))
        if "OTU" in name:
            otu_feature_idx.append(data.columns.get_loc(name))
    snp_data = data.iloc[:, snp_feature_idx]
    sel = SelectKBest(score_func=f_regression, k=1024)
    sel.fit(snp_data, y_label)
    pvalues = list(sel.pvalues_)

    otu_pvalues = []
    otu_data = data.iloc[:, otu_feature_idx[:500]]
    otu_feature = sm.add_constant(otu_data)
    lr_model = sm.OLS(y_label, otu_feature).fit()
    pvalues.extend(lr_model.pvalues.iloc[1:].to_list())
    otu_data = data.iloc[:, otu_feature_idx[500:]]
    otu_feature = sm.add_constant(otu_data)
    lr_model = sm.OLS(y_label, otu_feature).fit()
    pvalues.extend(lr_model.pvalues.iloc[1:].to_list())

    pvalues_idx = np.argsort(pvalues)
    if k is None:
        k = len(pvalues)
    select_features = np.asarray(snp_feature_idx + otu_feature_idx)[pvalues_idx[:k]]

    return select_features


def train(df_X: pd.DataFrame, df_Y: pd.DataFrame, train_test_ration=0.7):
    """start train"""
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(df_X),
        np.asarray(df_Y),
        train_size=train_test_ration,
        random_state=33
    )

    model_xgb(x_train, y_train, x_test, y_test)
    pass


def main_old():
    # snp_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\Genotype_on_phenotypes\827TSLW_SNP4"  # 需转为csv，读取xlsx巨巨巨慢
    snp_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\all_genotype data_test.csv"  # 需转为csv，读取xlsx巨巨巨慢
    assert os.path.exists(snp_path), f"{snp_path} is not exits."
    otu_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\微生物数据\核心OTU_827OTU0.7.csv"
    assert os.path.exists(otu_path), f"{otu_path} is not exits."
    label_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\谷子表型数据\Millet_12_phenotrypes.rename.csv"
    assert os.path.exists(label_path), f"{label_path} is not exits."
    save_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\data_process\test"

    dp = DataProcess()
    X_Y = dp.merge_data(snp_path, otu_path, label_path, save_path)
    exit()
    X_Y = pd.read_csv(os.path.join(save_path, "merge.csv"))
    X_Y = X_Y.set_index("Unnamed: 0", drop=True)
    Y = X_Y['TSLW']

    # select features
    feature_idx = []
    # for name in X_Y.columns.values:
    #     if "snp" in name:
    #         feature_idx.append(X_Y.columns.get_loc(name))
    if feature_idx:
        X = X_Y.iloc[:, feature_idx]
    else:
        X = X_Y.iloc[:, 12:]

    from sklearn.feature_selection import SelectKBest, f_regression
    X_fill = X.fillna(-1)
    # 选择特征基于F值
    sel = SelectKBest(score_func=f_regression, k=4096)
    sel.fit(X_fill, Y)
    select_idx = feature_selection_top_200(X_fill, Y, k=4096)
    X_1 = X.iloc[:, select_idx[:4096]]

    train(X_1, Y, train_test_ration=0.8)


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

    dp = DataProcess()
    df_merge = dp.read_data(os.path.join(save_path, "merge.csv"), header=0)
    df_merge = df_merge.set_index("Unnamed: 0", drop=True)
    label = df_merge['TSLW']

    X_snp_names = [name for name in df_merge.columns.values if "snp" in name]
    X_otu_names = [name for name in df_merge.columns.values if "OTU" in name]
    X_snp_pvalue = dp.f_regression(data=df_merge[X_snp_names], label=label)
    X_otu_pvalue = dp.linear_regression(data=df_merge[X_otu_names], label=label)
    pvalue_dict = {
        "name": X_snp_names + X_otu_names,
        "pvalue": X_snp_pvalue + X_otu_pvalue
    }
    pvalue_df = pd.DataFrame(pvalue_dict)
    r_order = pvalue_df.sort_values(by=["pvalue"])
    select_feature_names = list(r_order["name"][:256].values)

    X = df_merge[select_feature_names]
    trainer = Trainer()
    trainer.train(df_X=X, df_Y=label)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    random.seed(42)

    main()
    # dp = DataProcess()
