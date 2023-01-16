import time
#optimizer=keras.optimizers.SGD(0.01)
import tensorflow._api.v2.compat.v1 as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import datetime
import copy
import math
# import logging
from model import Net
from tensorflow import keras
from data_split import load_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.backend import set_session
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from collections import defaultdict
from skfuzzy.cluster import cmeans
# from keras.utils import to_categorical
from tensorflow.keras.utils import  to_categorical
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import confusion_matrix
# import  matplotlib.pyplot  as  plt
# import os
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
batch_size = 64
warnings.filterwarnings("ignore")
shards = []
def create_clients(image_list, label_list, num_clients, initial):
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    shards = []
    shards1 = []
    shards.append(shards1)
    shards2 = []
    shards.append(shards2)
    shards3 = []
    shards.append(shards3)
    shards4 = []
    shards.append(shards4)
    shards5 = []
    shards.append(shards5)
    shards6 = []
    shards.append(shards6)
    shards7 = []
    shards.append(shards7)
    shards8 = []


    shards.append(shards8)
    shards9 = []
    shards.append(shards9)
    shards10 = []
    shards.append(shards10)
    shards11 = []
    shards.append(shards11)
    shards12 = []
    shards.append(shards12)
    shards13 = []
    shards.append(shards13)
    shards14 = []
    shards.append(shards14)
    shards15 = []
    shards.append(shards15)
    shards16 = []
    shards.append(shards16)
    shards17 = []
    shards.append(shards17)
    shards18 = []
    shards.append(shards18)
    shards19 = []
    shards.append(shards19)
    shards20 = []
    shards.append(shards20)

    size = len(data)
    # shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    # shards = [data[i:i + size] for i in range(0, size)]
    #shards1 =np.array_split(data, 40)
    for n in range(len(data)):
        if n < len(data)/3:
            ran = np.random.randint(0,5)
            shards[ran].append(data[n])
        elif n < (2*len(data))/3 and n > len(data)/3:
            ran = np.random.randint(5, 15)
            shards[ran].append(data[n])
        else:
            ran = np.random.randint(15, 20)
            shards[ran].append(data[n])


    #for j in range(num_clients):
        #if j<8:
            #shards.append(shards1[j])
        #elif j>=8 and j<14:
            #shards.append(np.concatenate((shards1[j],shards1[j+1]),axis=0))
        #elif j>=14 and j<18:
            #shards.append(np.concatenate((shards1[j+6],shards1[j+7],shards1[j+8]),axis=0))
        #else:
            #shards.append(np.concatenate((shards1[j+14],shards1[j+15],shards1[j+16],shards1[j+17]),axis=0))


    #shards = np.array_split(data, num_clients)

    assert (len(shards) == len(client_names))
    return {client_names[i] : shards[i] for i in range(len(client_names))}


def batch_data(data, bs = batch_size):
    # data, label =zip(*data_shard)
    # data = np.array(data).astype(np.float)
    # label = np.array(label)
    # dataset = tf.data.Dataset.from_tensor_slices((np.array(data), np.array(label)))
    # dataset = dataset.shuffle(len(label)).batch(bs)

    # feature = np.array(data[0]).astype(np.float)
    # label = np.array(data[1])

    data, label =zip(*data)
    data = np.array(data).astype(np.float64)
    label = np.array(label)
    dataset = (data, label)
    datasets = dataset
    return datasets

# def init(model_):
#     model_.evaluate(x_test[0:1,...], y_test[0:1,...], verbose=0)

# def test_model(x_test , y_test, model, comm_round):
#     x_test = np.array(x_test)
#     y_test = np.array(y_test)
#     # cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     logits = model.predict(x_test)
#     loss = log_loss(y_test, logits)
#     acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(logits, axis=1))
#     print('comm_round: {} | global_acc: {: .3%} | global_loss: {}'.format(comm_round+1, acc, loss))
#
#     return acc,loss
def weight_scalling_factor(clients_trn_data, client_name_, num_epoch ):
    client_names1 = list(clients_trn_data.keys())
    client_names = client_names1[:num_clients-num_pass]
    # bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #
    # global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs

    ##把真正参与的client总的数据量求出来

    global_count = sum(clients_trn_data[client_name][0].shape[0] for client_name in client_names)
    local_count  = clients_trn_data[client_name_][0].shape[0]
    # if max_epoch > num_epoch:
    #   return ((local_count / global_count) * (10-10 * (math.exp(1/(num_epoch-max_epoch)))))
    # else:
    #     return (local_count / global_count) * 10
    return ((local_count / global_count) )

def weight_scalling_factor_demo(clients_trn_data ):
    client_names1 = list(clients_trn_data.keys())
    client_names = client_names1[:num_clients-num_pass]
    ##把真正参与的client总的数据量求出来
    global_count = sum(clients_trn_data[client_name][0].shape[0] for client_name in client_names)
    return  global_count

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
def train_cluster(x_train,y_train):
    n = [5]
    num = 15000
    p = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125])
    x_train = x_train.astype(np.float64)
    x_train_ex = x_train.T
    for n0 in n:
        center, u, u0, d, jm, p, fpc = cmeans(x_train_ex, c=n0, m=2, error=0.000001, maxiter=2000)
        # for delt in range(len(delta)):
        indexs = []
        Test_Data = []
        Test_Label = []
        final_pre_onehot = []
        final_label = []
        indexs1 = []
        Test_pre_onehot = []
        q = []

        for i in range(len(x_train)):
            u_list = []
            for s in range(n0):
                u_list.append(u[s][i])
            ordered_list = sorted(u_list, reverse=True)

            if abs(np.float64(ordered_list[0]) - np.float64(ordered_list[1])) < 0.2:  ## <0.25 (1 / (2 * n0))
                Test_Data.append(x_train[i])
                Test_Label.append(y_train[i])
                indexs1.append(i)

            else:
                indexs.append(i)

        a0 = []
        a1 = []
        a2 = []
        a3 = []
        a4 = []
        a5 = []
        a6 = []
        a7 = []
        a8 = []
        a9 = []
        a10 = []
        a11 = []
        a12 = []
        a13 = []
        a14 = []
        a15 = []
        a16 = []
        a17 = []
        a18 = []
        a19 = []
        a20 = []
        a21 = []
        a22 = []
        a23 = []
        a24 = []
        a25 = []
        a26 = []
        a27 = []
        a28 = []
        a29 = []
        a30 = []
        a31 = []
        a32 = []
        a33 = []
        a34 = []
        a35 = []
        a36 = []
        a37 = []
        a38 = []
        a39 = []
        a40 = []
        a41 = []
        a42 = []
        a43 = []
        a44 = []
        a45 = []
        a46 = []
        a47 = []
        a48 = []
        a49 = []
        a50 = []

        a = []
        ##取出n类中置信度最大的数
        for p in u:
            pre = np.argmax(u, axis=0)
        # ch_scores.append(calinski_harabasz_score(x_test, pre))
        # dbi_scores.append(davies_bouldin_score(x_test, pre))
        # sc_scores.append(silhouette_score(x_test, pre, sample_size=10000, metric='euclidean'))

        for ind in indexs:
            if pre[ind] == 0:
                a0.append(ind)
            elif pre[ind] == 1:
                a1.append(ind)
            elif pre[ind] == 2:
                a2.append(ind)
            elif pre[ind] == 3:
                a3.append(ind)
            elif pre[ind] == 4:
                a4.append(ind)
            elif pre[ind] == 5:
                a5.append(ind)
            elif pre[ind] == 6:
                a6.append(ind)
            elif pre[ind] == 7:
                a7.append(ind)
            elif pre[ind] == 8:
                a8.append(ind)
            elif pre[ind] == 9:
                a9.append(ind)
            elif pre[ind] == 10:
                a10.append(ind)
            elif pre[ind] == 11:
                a11.append(ind)
            elif pre[ind] == 12:
                a12.append(ind)
            elif pre[ind] == 13:
                a13.append(ind)
            elif pre[ind] == 14:
                a14.append(ind)
            elif pre[ind] == 15:
                a15.append(ind)
            elif pre[ind] == 16:
                a16.append(ind)
            elif pre[ind] == 17:
                a17.append(ind)
            elif pre[ind] == 18:
                a18.append(ind)
            elif pre[ind] == 19:
                a19.append(ind)
            elif pre[ind] == 20:
                a20.append(ind)
            elif pre[ind] == 21:
                a21.append(ind)
            elif pre[ind] == 22:
                a22.append(ind)
            elif pre[ind] == 23:
                a23.append(ind)
            elif pre[ind] == 24:
                a24.append(ind)
            elif pre[ind] == 25:
                a25.append(ind)
            elif pre[ind] == 26:
                a26.append(ind)
            elif pre[ind] == 27:
                a27.append(ind)
            elif pre[ind] == 28:
                a28.append(ind)
            elif pre[ind] == 29:
                a29.append(ind)
            elif pre[ind] == 30:
                a30.append(ind)
            elif pre[ind] == 31:
                a31.append(ind)
            elif pre[ind] == 32:
                a32.append(ind)
            elif pre[ind] == 33:
                a33.append(ind)
            elif pre[ind] == 34:
                a34.append(ind)
            elif pre[ind] == 35:
                a35.append(ind)
            elif pre[ind] == 36:
                a36.append(ind)
            elif pre[ind] == 37:
                a37.append(ind)
            elif pre[ind] == 38:
                a38.append(ind)
            elif pre[ind] == 39:
                a39.append(ind)
            elif pre[ind] == 40:
                a40.append(ind)
            elif pre[ind] == 41:
                a41.append(ind)
            elif pre[ind] == 42:
                a42.append(ind)
            elif pre[ind] == 43:
                a43.append(ind)
            elif pre[ind] == 44:
                a44.append(ind)
            elif pre[ind] == 45:
                a45.append(ind)
            elif pre[ind] == 46:
                a46.append(ind)
            elif pre[ind] == 47:
                a47.append(ind)
            elif pre[ind] == 48:
                a48.append(ind)
            elif pre[ind] == 49:
                a49.append(ind)
            elif pre[ind] == 50:
                a50.append(ind)

        # X_pca = PCA(n_components=3).fit_transform(x_test)
        # xs = X_pca[:, 0]
        # ys = X_pca[:, 1]
        # zs = X_pca[:, 2]
        # # cs = X_pca[:, 3]
        #
        # # X_pca = np.vstack((X_pca.T, y_test)).T
        # # df_pca = pd.DataFrame(X_pca,columns=['1st_Component','2st_Component', '3st_Component','label'])
        # # df_pca.head()
        # fig = plt.figure()
        # # plt.title("clusters_K=", n0)
        # ax = Axes3D(fig)
        # # ax.scatter(df_pca,hue='label', xs='1st_Component',ys='2st_Component',zs='3st_Component')
        # ax.scatter(xs, ys, zs, c=pre, marker='^')
        # # ax.scatter(xs, ys, c=pre, marker='^')
        # plt.show()
        _a_ = []
        _a_label = []
        # _a = [[]]
        if len(a0) != 0:
            a.append(a0)
        if len(a1) != 0:
            a.append(a1)
        if len(a2) != 0:
            a.append(a2)
        if len(a3) != 0:
            a.append(a3)
        if len(a4) != 0:
            a.append(a4)
        if len(a5) != 0:
            a.append(a5)
        if len(a6) != 0:
            a.append(a6)
        if len(a7) != 0:
            a.append(a7)
        if len(a8) != 0:
            a.append(a8)
        if len(a9) != 0:
            a.append(a9)
        if len(a10) != 0:
            a.append(a10)
        if len(a11) != 0:
            a.append(a11)
        if len(a12) != 0:
            a.append(a12)
        if len(a13) != 0:
            a.append(a13)
        if len(a14) != 0:
            a.append(a14)
        if len(a15) != 0:
            a.append(a15)
        if len(a16) != 0:
            a.append(a16)
        if len(a17) != 0:
            a.append(a17)
        if len(a18) != 0:
            a.append(a18)
        if len(a19) != 0:
            a.append(a19)
        if len(a20) != 0:
            a.append(a20)
        if len(a21) != 0:
            a.append(a21)
        if len(a22) != 0:
            a.append(a22)
        if len(a23) != 0:
            a.append(a23)
        if len(a24) != 0:
            a.append(a24)
        if len(a25) != 0:
            a.append(a25)
        if len(a26) != 0:
            a.append(a26)
        if len(a27) != 0:
            a.append(a27)
        if len(a28) != 0:
            a.append(a28)
        if len(a29) != 0:
            a.append(a29)
        if len(a30) != 0:
            a.append(a30)
        if len(a31) != 0:
            a.append(a31)
        if len(a32) != 0:
            a.append(a32)
        if len(a33) != 0:
            a.append(a33)
        if len(a34) != 0:
            a.append(a34)
        if len(a35) != 0:
            a.append(a35)
        if len(a36) != 0:
            a.append(a36)
        if len(a37) != 0:
            a.append(a37)
        if len(a38) != 0:
            a.append(a38)
        if len(a39) != 0:
            a.append(a39)
        if len(a40) != 0:
            a.append(a40)
        if len(a41) != 0:
            a.append(a41)
        if len(a42) != 0:
            a.append(a42)
        if len(a43) != 0:
            a.append(a43)
        if len(a44) != 0:
            a.append(a44)
        if len(a45) != 0:
            a.append(a45)
        if len(a46) != 0:
            a.append(a46)
        if len(a47) != 0:
            a.append(a47)
        if len(a48) != 0:
            a.append(a48)
        if len(a49) != 0:
            a.append(a49)
        if len(a50) != 0:
            a.append(a50)
        for i in range(len(a)):
            if len(a[i]) != 0:
                delda = []
                for k in range(len(a[i])):
                    u_list = []
                    for s in range(n0):
                        u_list.append(u[s][a[i][k]])
                    ordered_list = sorted(u_list, reverse=True)
                    delda.append(np.float64(ordered_list[0]) - np.float64(ordered_list[1]))
                max_indexs = []
                max_index = np.argsort(-np.array(delda))
                # max_index = delda.index(max(delda))
                if len(max_index) > num:
                    max_index = max_index[:num]
                    for l in max_index:
                        max_indexs.append(a[i][l])
                else:
                    for l in max_index:
                        max_indexs.append(a[i][l])
                _a = []
                _alabel = []
                for o in max_indexs:
                    _a.append(x_train[o])
                    _alabel.append(y_train[o])
                _a_.append(_a)
                _a_label.append(_alabel)
        final_pre = []
        final_rec = []
        final_score = []
        final_f1 = []
        TP = []
        TN = []
        FP = []
        FN = []
        predic_ = []
        # pre = []
        # labe = []
        # pre_ = []
        # labe_ = []
        # score_ = []
        x_clus = []
        y_clus = []
        for j in range(len(_a_)):
            if len(_a_[j]) != 0:
                for i in range(len(_a_[j])):
                    x_clus.append(_a_[j][i])
                    y_clus.append(_a_label[j][i])
    return x_clus,y_clus





#####RL

x_train, y_train, x_test, y_test = load_data()
start = time.clock()
x_clus,y_clus = train_cluster(x_train,y_train)
end = time.clock()
time1 = end-start
#y_train =np.array(y_train)


# x_test = np.array(x_test).astype(np.float)
y_test = np.array(y_test)
num_clients=20
x_test = np.array(x_test).reshape(-1, 11, 11, 1)
x_clus = np.array(x_clus).reshape(-1, 11, 11, 1)
print("训练数据量:", len(x_clus))
x_train = np.array(x_train).reshape(-1, 11, 11, 1)
y_clus = np.array(y_clus)
clients = create_clients(x_clus,y_clus, num_clients ,initial='client')
clients_batched = dict()



for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
#     clients_batched[client_name] = data
comms_round = 50
smlp_global = Net()
global_model = smlp_global.model_contruc()
#x_test = x_test.astype(np.float64)
#x_test_ex = x_test.T
max_epoch = 1
# n = [2]
# n= [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13, 14,15, 16,18,20]
glob_acc = []
glob_loss = []
all_acc = []
all_loss = []
acc = []
epoch_scaled = []
Q = 15

p = np.array([0.5,0.25,0.125,0.0625,0.03125,0.03125])
#x_train = np.array(x_train).reshape(-1, 11, 11, 1)
# thetas= []
# upload_clients = []
# upload_clients_datanum = []
# upload_clients_weights = []
# select_Q_weights = []
# select_Q_datanum = []
for comm_round in range(comms_round):
    if comm_round ==0:
        start = time.clock()

    # np.random.seed(0)
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
    if comm_round <2:
        # num_pass = random.randint(25, 29)
        num_pass = 0
    else:
        # num_pass = np.random.choice([0,1,2,3,4,5], p=p.ravel())
        # num_pass = random.randint(25, 29)
        num_pass = 0
    for client in client_names:
        if real_client < (num_clients-num_pass):
            model_ = Net()
            _model = model_.model_contruc()
            # init(model)
            global_weights = global_model.get_weights()
            _model.set_weights(global_weights)
            feature = clients_batched[client][0]
            label = clients_batched[client][1]
            # for (client, data) in clients_batched.items():

            # num_epoch = random.randint(1, max_epoch)
            # print(client, "epoch =",num_epoch)
            num_epoch = 1
            history = _model.fit(feature,label, epochs=1, verbose=1, batch_size=32)

            acc_num = history.history["acc"]
            loss_num = history.history["loss"]
            theta = 0.6*np.exp(-float(history.history["loss"][-1]))+0.4*np.exp(-len(feature)/weight_scalling_factor_demo(clients_batched))
            # theta = 0.6 * np.exp(-float(history.history["loss"][-1])) + 0.4 * np.exp(-len(feature) )
            # print(theta)
            if theta > 0.4:
                upload_clients.append(client)
                thetas.append(theta)
                upload_clients_datanum.append(float(len(feature)))
                upload_clients_weights.append(_model.get_weights())

            acc_client.append(acc_num[-1] * (float(len(feature))))
            loss_client.append(loss_num[-1] * (float(len(feature))))

            real_client = real_client+1

            # score = _model.evaluate(x_test, y_test)
            # score_train_local = _model.evaluate(feature, label)
            # print(client, 'accuracy is', score_train_local[1])
            # print(client, 'accuracy is', score[1])

            scaling_factor = weight_scalling_factor(clients_batched, client, num_epoch )
            scaling_weights = scale_model_weights(_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaling_weights)
            epoch_scaled.append(scaling_factor)

            # tf.keras.backend.clear_session()
            # tf.reset_default_graph()

    thetas_index = np.argsort(-np.array(thetas))
    if len(thetas)> Q:
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
    print('cluser_time = ', time1)
    if comm_round == 0:
        end = time.clock()
        time2 = end - start
        time = time1 + time2
        print('time = ', time)
    score_train = global_model.evaluate(x_train, y_train)
    #History = global_model.fit(x_train, y_train, batch_size=32)
    # print('comm_round: {} '.format(comm_round + 1))
    print("All_data_acc=", score_train[1], "All_data_loss=", score_train[0])
    print('comm_round: {} | global_acc: {: .3%} | global_loss: {}'.format(comm_round + 1, acc_sum, loss_sum))

    all_acc.append(score_train[1])
    all_loss.append(score_train[0])
    glob_acc.append(acc_sum)
    glob_loss.append(loss_sum)
    #print("All_data_acc=", History.history["acc"], "All_data_loss=", History.history["loss"])

    outcom1 = []
    outcom2 = []
    outcom3 = []
    outcom4 = []
    outcom5 = []



    if comm_round!=0 and comm_round % 4== 0:
        y_pred = global_model.predict(x_test, batch_size=64)
        acc_score_ = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        pre_score_ = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        recall_score_ = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        f1_score_ = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print( '在全局模型上测试准确率为：', acc_score_)
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