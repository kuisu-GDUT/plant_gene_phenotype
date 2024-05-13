import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance, plot_tree

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# train_orig = pd.read_csv('train.csv', index_col='row_id', parse_dates=['time'])
# train_orig.head()
#
# X_tr, X_va = train.loc[train_idx, features], train.loc[val_idx, features]
# y_tr, y_va = train.loc[train_idx, 'congestion'], train.loc[val_idx, 'congestion']


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


def model_xgb(x_train, y_train, x_test, y_test):
    # fig, ax = plt.subplots(figsize=(15, 15))#设置图像大小
    # xgboost
    ### 训练模型
    model = XGBRegressor(
        learning_rate=0.2,
        n_estimators=100,  # 树的个数--100棵树建立xgboost
        max_depth=5,  # 树的深度
        min_child_weight=2,  # 叶子节点最小权重
        gamma=0.4,  # 惩罚项中叶子结点个数前的参数
        subsample=0.7,  # 随机选择70%样本建立决策树
        colsample_bytree=0.7,  # 随机选择70%特征建立决策树
        objective='reg:squarederror',  # 使用平方误差作为损失函数
        scale_pos_weight=1,  # 解决样本个数不平衡的问题
        random_state=1,  # 随机数
        reg_alpha=2,
        reg_lambda=2
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
              # early_stopping_rounds=10,
              verbose=True)

    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    print('mse:', mean_squared_error(predictions, y_test))
    print('rmse:', np.sqrt(mean_squared_error(predictions, y_test)))
    print('mae:', mean_absolute_error(predictions, y_test))
    # print('r2:', r2_score(predictions, y_test))

    return model


def data_process_with_combine(snp_path: str, otu_path: str, label_path: str, save_path: str):
    df_snp = pd.read_csv(snp_path, header=0)
    df_snp_names = ["snp_{}".format(i) for i in df_snp["ID"]]
    df_snp = df_snp.iloc[:, 9:]
    df_snp = df_snp.T
    df_snp.columns = df_snp_names
    df_snp = feature_selection(df_snp, threshold=0.5)
    df_snp.to_csv(os.path.join(save_path, "snp.csv"))

    df_otu = pd.read_csv(otu_path, header=0)
    df_otu = df_otu.set_index("ID", drop=True)
    df_otu = feature_selection(df_otu, threshold=0.5)
    df_otu.to_csv(os.path.join(save_path, "otu.csv"))
    # df_otu = (df_otu - df_otu.min()) / (df_otu.max() - df_otu.min()) #即简单实现标准化

    df_label = pd.read_csv(label_path, header=0)
    df_label_names = [i for i in df_label["Plant_traits"]]
    df_label = df_label.iloc[:, 1:]
    df_label = df_label.T
    df_label.columns = df_label_names
    df_label.to_csv(os.path.join(save_path, "label.csv"))

    x_data = pd.merge(df_snp, df_otu, how='left', left_index=True, right_index=True)
    Y_X = pd.merge(df_label, x_data, how="left", left_index=True, right_index=True)

    Y_X.to_csv(os.path.join(save_path, "merge.csv"))
    print(f"Y_X shape:{Y_X.shape}")
    print(f"Y_X info: {Y_X.info}")
    return Y_X


def feature_selection(data: pd.DataFrame, threshold=0.5):
    """删除方差阈值 < 0.8"""
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=threshold)
    print(f"before features: {data.shape}")
    sel.fit(data)
    data = data.iloc[:, sel.get_support(True)]
    print(f"after features: {data.shape}")
    print('Variances is %s' % sel.variances_)
    return data


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


def main():
    snp_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\基因型数据\all_genotype data.csv"  # 需转为csv，读取xlsx巨巨巨慢
    assert os.path.exists(snp_path), f"{snp_path} is not exits."
    otu_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\微生物数据\核心OTU_827OTU0.7.csv"
    assert os.path.exists(otu_path), f"{otu_path} is not exits."
    label_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\谷子表型数据\Millet_12_phenotrypes.rename.csv"
    assert os.path.exists(label_path), f"{label_path} is not exits."
    save_path = r"D:\03_data\11_MAT_DATA\06_data\谷子\data_process"

    # X_Y = data_process_with_combine(snp_path, otu_path, label_path, save_path)
    # exit()
    X_Y = pd.read_csv(os.path.join(save_path, "merge.csv"))
    X_Y = X_Y.set_index("Unnamed: 0", drop=True)
    Y = X_Y.iloc[:, 0]
    X = X_Y.iloc[:, 12:]
    X = X.fillna(-1)

    from sklearn.feature_selection import SelectKBest, f_regression
    # 选择特征基于F值
    sel = SelectKBest(score_func=f_regression, k=128)
    sel.fit(X, Y)
    X_1 = X.iloc[:, sel.get_support(True)]

    train(X_1, Y, train_test_ration=0.7)

    pass


if __name__ == '__main__':
    random.seed(42)

    main()
