from skfuzzy import cmeans
import numpy as np


def train_cluster(x_train, y_train):
    """
    将全集数据抽样
    :param x_train: 训练集输入
    :param y_train: 训练集输出
    :return:
    """
    n = 5
    num = 15000
    x_train = x_train.astype(np.float64)
    x_train_ex = x_train.T
    center, u, u0, d, jm, p, fpc = cmeans(x_train_ex, c=n, m=2, error=0.0001, maxiter=2000)

    # 取出n类中置信度最大的数
    for _ in u:
        label = np.argmax(u, axis=0)

    reliable_data = [[] for _ in range(n)]

    for i in range(len(x_train)):
        u_list = []
        for s in range(n):
            u_list.append(u[s][i])
        ordered_list = sorted(u_list, reverse=True)

        if abs(np.float64(ordered_list[0]) - np.float64(ordered_list[1])) >= 0.2:  # 最大
            reliable_data[label[i]].append(i)

    _a_ = []
    _a_label = []

    for i in range(len(reliable_data)):
        if len(reliable_data[i]) != 0:
            delda = []
            for k in range(len(reliable_data[i])):
                u_list = []
                for s in range(n):
                    u_list.append(u[s][reliable_data[i][k]])
                ordered_list = sorted(u_list, reverse=True)
                delda.append(np.float64(ordered_list[0]) - np.float64(ordered_list[1]))
            max_indexs = []
            max_index = np.argsort(-np.array(delda))
            # max_index = delda.index(max(delda))
            if len(max_index) > num:
                max_index = max_index[:num]
                for l in max_index:
                    max_indexs.append(reliable_data[i][l])
            else:
                for l in max_index:
                    max_indexs.append(reliable_data[i][l])
            _a = []
            _alabel = []
            for o in max_indexs:
                _a.append(x_train[o])
                _alabel.append(y_train[o])
            _a_.append(_a)
            _a_label.append(_alabel)
    x_clus = []
    y_clus = []
    for j in range(len(_a_)):
        if len(_a_[j]) != 0:
            for i in range(len(_a_[j])):
                x_clus.append(_a_[j][i])
                y_clus.append(_a_label[j][i])
    return x_clus, y_clus
