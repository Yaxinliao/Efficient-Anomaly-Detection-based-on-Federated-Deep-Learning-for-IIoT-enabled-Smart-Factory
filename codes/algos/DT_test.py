import numpy as np
import tensorflow._api.v2.compat.v1 as tf
# mpl.rcParams['font.sans-serif'] = [u'simHei']
# mpl.rcParams['axes.unicode_minus'] = False
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import codes.utils as utils


def main(data_train_x, data_train_y, data_test_x, data_test_y):
    print("dt:")
    tf.compat.v1.disable_eager_execution()
    tf.disable_v2_behavior()

    data_train_y = np.argmax(data_train_y, axis=1)
    data_test_y = np.argmax(data_test_y, axis=1)

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random')  # 实例化
    clf = clf.fit(data_train_x, data_train_y)  # 训练

    test_pre = clf.score(data_test_x, data_test_y)

    y_pred = clf.predict(data_test_x)
    acc_score_ = accuracy_score(data_test_y, y_pred)
    pre_score_ = precision_score(data_test_y, y_pred)
    recall_score_ = recall_score(data_test_y, y_pred)
    f1_score_ = f1_score(data_test_y, y_pred)
    print('在全局模型上测试准确率为：', acc_score_)
    print('在全局模型上测试precision为：', pre_score_)
    print('在全局模型上测试recall_score_为：', recall_score_)
    print('在全局模型上测试f1_score_为：', f1_score_)

    print('测试集打分', test_pre)
    print('训练集打分', clf.score(data_train_x, data_train_y))

    # 测试集打分 0.9034707820914717
    # 训练集打分 0.999890691578191
