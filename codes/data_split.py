import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import random,csv
import numpy as np
# def next_batch(feature_list,label_list,size):
#     feature_batch_temp=[]
#     label_batch_temp=[]
#     f_list = random.sample(range(len(feature_list)), size)
#     for i in f_list:
#         feature_batch_temp.append(feature_list[i])
#     for i in f_list:
#         label_batch_temp.append(label_list[i])
#     return feature_batch_temp,label_batch_temp

def Add_noise(x, d, SNR):
    P_signal=np.sum(abs(x)**2)
    P_d=np.sum(abs(d)**2)
    P_noise=P_signal/10**(SNR/10)
    noise=np.sqrt(P_noise/P_d)*d
    noise_signal = x + noise
    return noise_signal
#def wgn(x, snr):
  #P_signal = np.sum(abs(x)**2)/len(x)
  #P_noise = P_signal/10**(snr/10.0)
  #return np.random.randn(len(x)) * np.sqrt(P_noise)
def wgn(x, snr):
    #batch_size, len_x = x.shape
    Ps = np.sum(abs(x)**2)
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    return x + noise



def load_data():
    global feature
    global label
    global feature_full
    global label_full
    mu = 0
    sigma = 0.6
    clients_ID=[]
    clients_label_ID=[]
    feature=[]
    label=[]
    feature_full=[]
    label_full=[]
    snr = 5
    #x = []
    # file_path = "C:\\Users\\LYX\\Desktop\\pythonProject\\KDDtest_onehot.csv"
    # file_path ="D:\\file\python_program\pythonProject\\KDDtest_onehot - 副本.csv"
    #file_path = "C:\\Users\\LYX\\Desktop\\pythonProject\\KDDTest+_数值化_标准化_归一化_3+6独热结果"
    file_path = "C:\\Users\\LYX\\Desktop\\pythonProject\\test_3+6项独热编码.csv"
    # file_path = "C:\\Users\\LYX\\Desktop\\kddcupcorrcted_3+6独热编码结果fina.csv"
    ##file_path = "C:\\Users\\LYX\\Desktop\\pythonProject\\KDDTest+_数值化_标准化_归一化_3+6独热结果.csv"

    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            #print (i)
            label_list=[0]*2
            #noise = np.random.randn(1,121)  # 产生N(0,1)噪声数据
            #noise = noise - np.mean(noise)
            x = np.array(i[:121])
            x = x.astype(float)
            #x_noise = wgn(x,snr)
            #feature.append(x_noise)
            #for j in range(len(i) - 1):
                #i[j] = float(i[j])
                #i[j] += random.gauss(mu, sigma)
            feature.append(x)      #121维特征数据
            label_list[int(i[121])] = 1     #最后标签数据
            label.append(label_list)

    # file_path_full = "C:\\Users\\LYX\\Desktop\\pythonProject\\KDD_train_onehot.csv"
    # file_path_full ="D:\\file\python_program\pythonProject\\KDD_train_onehot - 副本.csv"
    # file_path_full = "D:\\onedrive\\onedrive-bupt\\OneDrive - bupt.edu.cn\\file\\previous\\python files/chapter_2/KDD_train_onehot.csv"
    file_path_full = "C:\\Users\\LYX\\Desktop\\pythonProject\\train_3+6项独热编码.csv"
    # file_path_full = "C:\\Users\\LYX\\Desktop\\pythonProject\\test_3+6项独热编码.csv"
    ##file_path_full = "C:\\Users\\LYX\\Desktop\\pythonProject\\KDDTrain+_数值化_标准化_归一化_3+6独热结果.csv"
    # file_path_full = "C:\\Users\\LYX\\Desktop\\corrcted_3+6独热编码结果_fina.csv"


    with (open(file_path_full,'r')) as data_from_full:
        csv_reader_full=csv.reader(data_from_full)
        for j in csv_reader_full:
            # print i
            label_list_full=[0]*2
            x = np.array(j[:121])
            x = x.astype(float)
            #x_noise = wgn(x, snr)

            feature_full.append(x)
            label_list_full[int(j[121])] = 1
            label_full.append(label_list_full)
    feature_test = feature
    feature_train = feature_full
    label_test = label
    label_train = label_full

    label_train = np.array(label_train)
    label_test = np.array(label_test)

    data_ = list(zip(feature_train, label_train))
    random.shuffle(data_)
    data = random.sample(data_, 10000)
    x_test1, y_test1 = zip(*data)

    x_train = np.array(feature_train)
    #x_train = np.array(feature_train).reshape(-1, 11, 11, 1)

    y_train = label_train
    y_test = label_test

    x_test = np.array(feature_test)

    # x_test = np.array(x_test).astype(np.float64)
    # y_test = np.array(y_test)

    # x_train_scaled = np.array(x_train).reshape(-1, 11, 11, 1)
    # # x_valid_scaled = np.array(x_valid).reshape(-1, 11, 11, 1)
    # x_test_scaled = np.array(feature_test).reshape(-1, 11, 11, 1)
    # x_train_scaled = np.array(x_train)
    # x_valid_scaled = np.array(x_valid).reshape(-1, 11, 11, 1)
    # x_test_scaled = np.array(feature_test)
    # xdata_split = np.array_split(x_train,num_clients)
    # ydata_split = np.array_split(y_train,num_clients)
    # for i in range(num_clients):
    #
    #     clients_ID[i] =np.array(xdata_split[i]).reshape(-1, 11, 11, 1)
    #     clients_label_ID[i] =ydata_split[i]
    return x_train, y_train, x_test, y_test
    # train_clients_data =


