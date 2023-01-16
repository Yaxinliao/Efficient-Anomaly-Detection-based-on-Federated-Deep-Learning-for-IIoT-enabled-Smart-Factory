import random
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# optimizer=keras.optimizers.SGD(0.01)
import tensorflow._api.v2.compat.v1 as tf
from skfuzzy.cluster import cmeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import codes.utils as utils

# from keras.utils import to_categorical
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# import  matplotlib.pyplot  as  plt
# import os
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
tf.enable_eager_execution()
batch_size = 64
warnings.filterwarnings("ignore")
shards = []


def create_clients(image_list, label_list, num_clients, initial):
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    shards = [[] for _ in range(num_clients)]
    random.seed(4)
    a = random.sample(range(len(data) / (2*num_clients), num_clients), num_clients-1)
    a.sort()

    for i in range(len(a)):
        if i == 0:
            shards[0] = data[0:a[0]]
        else:
            shards[i] = data[a[i-1]:a[i]]
    shards[num_clients-1] = data[a[num_clients-2]:len(data)]

    assert (len(shards) == len(client_names))
    return {client_names[i]: shards[i] for i in range(len(client_names))}


def batch_data(data, bs=batch_size):
    data, label = zip(*data)
    data = np.array(data).astype(np.float64)
    label = np.array(label)
    dataset = (data, label)
    datasets = dataset
    return datasets


def weight_scalling_factor(clients_trn_data, client_name_, num_epoch, num_clients, num_pass):
    client_names1 = list(clients_trn_data.keys())
    client_names = client_names1[:num_clients - num_pass]
    # bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #
    # global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs

    ##把真正参与的client总的数据量求出来

    global_count = sum(clients_trn_data[client_name][0].shape[0] for client_name in client_names)
    local_count = clients_trn_data[client_name_][0].shape[0]
    # if max_epoch > num_epoch:
    #   return ((local_count / global_count) * (10-10 * (math.exp(1/(num_epoch-max_epoch)))))
    # else:
    #     return (local_count / global_count) * 10
    return ((local_count / global_count))


def weight_scalling_factor_demo(clients_trn_data, num_clients, num_pass):
    client_names1 = list(clients_trn_data.keys())
    client_names = client_names1[:num_clients - num_pass]
    ##把真正参与的client总的数据量求出来
    global_count = sum(clients_trn_data[client_name][0].shape[0] for client_name in client_names)
    return global_count


def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)

    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    # scaled_weight_list = np.array(scaled_weight_list)
    # layer_mean = scaled_weight_list.mean(axis=0)
    layer_mean = np.sum(scaled_weight_list, axis=0)

    return layer_mean


def main(x_train, y_train, x_test, y_test):
    print("fcm:")
    start = time.perf_counter()
    x_clus, y_clus = utils.train_cluster(x_train, y_train)  # 抽样
    end = time.perf_counter()
    time1 = end - start
    print('cluser_time = ', time1)

    y_test = np.array(y_test)
    num_clients = 20
    x_test = np.array(x_test).reshape(-1, 11, 11, 1)  # 121 维整理成 1*11*11*1 维
    x_clus = np.array(x_clus).reshape(-1, 11, 11, 1)
    print("训练数据量:", len(x_clus))
    x_train = np.array(x_train).reshape(-1, 11, 11, 1)
    y_clus = np.array(y_clus)
    clients = create_clients(x_clus, y_clus, num_clients, initial='client')
    clients_batched = dict()

    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data)
    comms_round = 50
    smlp_global = utils.Net()
    global_model = smlp_global.model_contruc()

    glob_acc = []
    glob_loss = []
    all_acc = []
    all_loss = []
    acc = []
    epoch_scaled = []
    Q = 15

    for comm_round in range(comms_round):
        if comm_round == 0:
            start = time.perf_counter()

        scaled_local_weight_list = list()
        thetas = []
        upload_clients = []
        upload_clients_datanum = []
        upload_clients_weights = []
        select_Q_weights_list = list()
        select_Q_weights = []
        select_Q_datanum = []
        loss_client = list()
        acc_client = list()
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)
        real_client = 0
        num_pass = 0
        for client in client_names:
            if real_client < (num_clients - num_pass):
                model_ = utils.Net()
                _model = model_.model_contruc()
                # init(model)
                global_weights = global_model.get_weights()
                _model.set_weights(global_weights)
                feature = clients_batched[client][0]
                label = clients_batched[client][1]
                num_epoch = 1
                history = _model.fit(feature, label, epochs=1, verbose=1, batch_size=32)

                acc_num = history.history["accuracy"]
                loss_num = history.history["loss"]
                theta = 0.6 * np.exp(-float(history.history["loss"][-1])) + 0.4 * np.exp(
                    -len(feature) / weight_scalling_factor_demo(clients_batched, num_clients, num_pass))
                # theta = 0.6 * np.exp(-float(history.history["loss"][-1])) + 0.4 * np.exp(-len(feature) )
                # print(theta)
                if theta > 0.4:
                    upload_clients.append(client)
                    thetas.append(theta)
                    upload_clients_datanum.append(float(len(feature)))
                    upload_clients_weights.append(_model.get_weights())

                acc_client.append(acc_num[-1] * (float(len(feature))))
                loss_client.append(loss_num[-1] * (float(len(feature))))

                real_client = real_client + 1

                scaling_factor = weight_scalling_factor(clients_batched, client, num_epoch, num_clients, num_pass)
                scaling_weights = scale_model_weights(_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaling_weights)
                epoch_scaled.append(scaling_factor)

        thetas_index = np.argsort(-np.array(thetas))
        if len(thetas) > Q:
            for i in range(Q):
                select_Q_datanum.append(upload_clients_datanum[thetas_index[i]])
            for i in range(Q):
                factor = float(upload_clients_datanum[thetas_index[i]]) / float(sum(select_Q_datanum))
                select_Q_weights_list.append((pd.Series(upload_clients_weights[thetas_index[i]]) * factor).tolist())
            average_weights = sum_scaled_weights(select_Q_weights_list).tolist()
            print('real_client = ', Q)
        else:
            for i in range(len(thetas)):
                select_Q_datanum.append(upload_clients_datanum[thetas_index[i]])
            for i in range(len(thetas)):
                factor = float(upload_clients_datanum[thetas_index[i]]) / float(sum(select_Q_datanum))
                select_Q_weights_list.append((pd.Series(upload_clients_weights[thetas_index[i]]) * factor).tolist())
            average_weights = sum_scaled_weights(select_Q_weights_list).tolist()
            print('real_client = ', len(thetas))

        global_model.set_weights(average_weights)
        global_weights_ = global_model.get_weights()
        loss_sum = sum(loss_client) / float(len(y_clus))
        acc_sum = sum(acc_client) / float(len(y_clus))
        print('训练结果*****')
        if comm_round == 0:
            end = time.perf_counter()
            time2 = end - start
            timeSum = time1 + time2
            print('time = ', timeSum)
        score_train = global_model.evaluate(x_train, y_train)
        # History = global_model.fit(x_train, y_train, batch_size=32)
        # print('comm_round: {} '.format(comm_round + 1))
        print("All_data_acc=", score_train[1], "All_data_loss=", score_train[0])
        print('comm_round: {} | global_acc: {: .3%} | global_loss: {}'.format(comm_round + 1, acc_sum, loss_sum))

        all_acc.append(score_train[1])
        all_loss.append(score_train[0])
        glob_acc.append(acc_sum)
        glob_loss.append(loss_sum)
        # print("All_data_acc=", History.history["acc"], "All_data_loss=", History.history["loss"])

        outcom1 = []
        outcom2 = []
        outcom3 = []
        outcom4 = []
        outcom5 = []

        if comm_round != 0 and comm_round % 4 == 0:
            y_pred = global_model.predict(x_test, batch_size=64)
            acc_score_ = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            pre_score_ = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            recall_score_ = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            f1_score_ = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            print('在全局模型上测试准确率为：', acc_score_)
            print('在全局模型上测试precision为：', pre_score_)
            print('在全局模型上测试recall_score_为：', recall_score_)
            print('在全局模型上测试f1_score_为：', f1_score_)
            outcom1.append(acc_score_)
            outcom2.append(pre_score_)
            outcom3.append(recall_score_)
            outcom4.append(f1_score_)

    fig2 = plt.figure()
    x_ = range(100)
    y_ = np.array(acc)
    # plt.axis([0, 0.5, 0, 1])
    plt.grid
    plt.plot(x_, y_)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.title('loss随通信轮次变化')
    plt.show()
