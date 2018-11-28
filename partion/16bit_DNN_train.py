# -*- coding: utf-8 -*-
"""
keras配置网络
8bit输入，8bit输出，训练之后的网络存在固定的文件

不再是一个极化码的译码器，而是一个随机码的对应+，=极化码结构的译码器
误码率的统计也变了
测试数据的产生也变了
和之前戴彬的版本已经没有任何相似之处了

训练的信噪比是4,5,6dB，而测试则是5dB的信噪比
by woodencat
"""

from __future__ import print_function, division

import tensorflow as tf
import numpy as np  # NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库
import keras
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda
from keras.optimizers import RMSprop, Adam
from keras.layers import SimpleRNN, LSTM, GRU, Activation
from keras.layers.wrappers import Bidirectional
from keras import backend as K
import time

from shutil import copyfile
import matplotlib.pyplot as plt
import sys
import math  # 支持一些数学函数，以及特定的数学变量

# 不加这个会报错，提醒我电脑CPU太差了，不能用这个破CPU运行tensorflow....(╬￣皿￣)=○.....(╬￣皿￣)=○
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 这样报错只出现warning和error，设置为1就什么信息都显示


# ################################################################
# 极化码编码参数
code_k = 8  # 信息位码长
code_n = 8  # 总的码长，可以看出来码率0.5
code_rate = 1.0 * code_k / code_n  # 算码率，有一个浮点数，最后结果就是浮点数了
word_seed = 786000
noise_seed = 345000

# 训练信噪比
start_snr = 3
stop_snr = 7
scaling_factor = np.arange(start_snr, stop_snr + 1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# 测试信噪比序列
validation_start_snr = 3
validation_stop_snr = 7
validation_snr = np.arange(validation_start_snr, validation_stop_snr + 1, 1,dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组
batch_size_validation = 16  # 用于验证的码字有这么多一组
# ########### Neural network config####################

epochnum = 2 ** code_k   # 每次训练这么多组code_n bit的码字，必须为2**code_k
batch_size = epochnum * len(scaling_factor)  # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！
batch_in_epoch = 100  # 每训练这么多次有一波计算误码率的操作
batches_for_val = 10  # 貌似使用这个来计算误帧率,要有多个帧才能计算误帧率
num_of_batch = 500  # 取名有些混乱，这个是训练的次数
LEARNING_RATE = 0.0001  # 学习率 不设置的话函数自动默认是0.001
train_on_zero_word = False
test_on_zero_word = False
load_weights = False
is_training = True

wordRandom = np.random.RandomState(word_seed)  # 伪随机数产生器，（seed）其中seed的值相同则产生的随机数相同
random = np.random.RandomState(noise_seed)

# 手动注释代码设置网络
nn_set = 'DNN'
f = open('16bit_DNN.txt', 'w')  # 开个文件，记录测试的误码率
print('当前配置的神经网络是： ', nn_set)


# #######################################################

# 定义各种小函数
def bitrevorder(x):
    m = np.amax(x)  # 输入的数组里最大的数
    n = np.ceil(np.log2(m)).astype(int)  # 这个最大的数用二进制表示有n位
    for i in range(0, len(x)):  # i从0到len(x),这个序列有i位，都要反转
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)  # int将字符串转为整形，这里是转为2进制整形。[::-1]序列倒序输出
    return x  # str.format 把format里的东西放在str对应的位置，例如："Hello, {0} and {1}!".format("John", "Mary")


def polar_design_awgn(N, k, snr_dB):
    S = 10 ** (snr_dB / 10)  # 计算信噪比公式，10log10S/N，反着来，得到信息值
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)  # 高斯密度进化，选出合格的信道
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))
    A = np.zeros(N, dtype=bool)
    A[idx] = True
    # print(A)
    return A


def full_adder(a, b, c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s, c


def add_bool(a, b):
    k = len(a)
    s = np.zeros(k, dtype=bool)
    c = False
    for i in reversed(range(0, k)):
        s[i], c = full_adder(a[i], b[i], c)
    return s


def inc_bin(a):
    k = len(a)
    increment = np.hstack((np.zeros(k - 1, dtype=int), np.ones(1, dtype=int)))
    a = add_bool(a, increment)
    return a


def polar_transform_iter(u):  # encoding
    N = len(u)  # 返回对象长度
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):  # 编码做log2N层的运算，每层都把这N个元素处理个遍
        i = 0
        while i < N:  # i是N个元素中的第i个
            for j in range(0, n):  # 每轮的步长是n，j表示n步长内的操作
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]  # ^按位异或
            i = i + 2 * n
        n = 2 * n  # 步长n逐层翻倍
    return x


# Data Generation
def create_mix_epoch(code_k, code_n, scaling_factor, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1, code_n], dtype=np.float32)
    Y = np.zeros([1, code_n], dtype=np.int64)

    # code_k个信息位，所有出现的0,1组合一共有numofcode_n个
    numofcode_n = 2 ** code_k
    x = np.zeros([numofcode_n, code_n], dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    d = np.zeros([numofcode_n, code_k], dtype=np.int64)
    for sf_i in scaling_factor:
        A = polar_design_awgn(code_n, code_k, sf_i)  # A是bool型的玩意，来判断这个信道是不是合适传输的
        # print("A是这个东西", A)
        # #### 在这里加入循环！！！！！！！！！！！！！！
        if is_zeros_word:  # 用全0数据训练
            d = 0 * wordRandom.randint(0, 2, size=(numofcode_n, code_k))  # max取值只能到2，不能到1
        else:
            # 把d变成1，2,3,4,5然后转化为2进制，从而遍历所有的情况，看看是不是我的网络设置有毛病
            for k in range(1, numofcode_n):  # 在码长固定的情况下遍历所有的可能情况
                d[k] = inc_bin(d[k - 1])

        for i in range(0, numofcode_n):
            x[i] = polar_transform_iter(d[i])

        snr_lin = 10.0 ** (sf_i / 10.0)
        noise = np.sqrt(1.0 / (2.0 * snr_lin * code_rate))
        X_p_i = random.normal(0.0, 1.0, x.shape) * noise + (1) * (1 - 2 * x)  # random.normal按照正态分布取随机数
        # X_p_i = random.normal(0.0, 1.0, x.shape) * noise + x  # random.normal按照正态分布取随机数
        x_llr_i = (1 - X_p_i) / 2
        # x_llr_i = 2 * X_p_i / (noise ** 2)
        X = np.vstack((X, x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
        Y = np.vstack((Y, d))  # u是单纯的原始码

    X = X[1:]  # X是编码加噪声后接收端处理过的对数似然信息
    Y = Y[1:]  # Y是最初未编码的0,1信息
    return X, Y


# 在一个固定信噪比下，产生numOfWordSim个码长为code_n的极化码
def create_mix_epoch_validation(code_k, code_n, numOfWordSim, validation_snr, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1, code_n], dtype=np.float32)
    Y = np.zeros([1, code_n], dtype=np.int64)

    x = np.zeros([numOfWordSim, code_n], dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    u = np.zeros([numOfWordSim, code_n], dtype=np.int64)
    d = np.zeros([numOfWordSim, code_k], dtype=np.int64)
    A = polar_design_awgn(code_n, code_k, validation_snr)  # A是bool型的玩意，来判断这个信道是不是合适传输的
    # print("A是这个东西", A)
    # #### 在这里加入循环！！！！！！！！！！！！！！
    if is_zeros_word:  # 用全0数据训练
        d = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
    else:
        d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # 随机生成训练数据

    u[:, A] = d
    for i in range(0, numOfWordSim):
        x[i] = polar_transform_iter(u[i])

    snr_lin = 10.0 ** (validation_snr / 10.0)
    noise = np.sqrt(1.0 / (2.0 * snr_lin * code_rate))
    X_p_i = random.normal(0.0, 1.0, x.shape) * noise + (1) * (1 - 2 * x)  # random.normal按照正态分布取随机数
    # X_p_i = random.normal(0.0, 1.0, x.shape) * noise + x  # random.normal按照正态分布取随机数
    x_llr_i = (1 - X_p_i) / 2
    # x_llr_i = 2 * X_p_i / (noise ** 2)
    X = np.vstack((X, x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
    Y = np.vstack((Y, u))  # u是单纯的原始码

    X = X[1:]  # X是编码加噪声后接收端处理过的对数似然信息
    Y = Y[1:]  # Y是最初未编码的0,1信息
    return X, Y


# 这个东西还是有点毛病，这写得可读性太差了！
def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    for i in range(0, snr_db.shape[0]):
        A = polar_design_awgn(code_n, code_k, snr_db[i])
        Y_v_pred_i = Y_v_pred[i*numOfWordSim:(i+1)*numOfWordSim,A]
        Y_v_i = Y_v[i*numOfWordSim:(i+1)*numOfWordSim,A]
        ber_test[i] = np.abs(((Y_v_pred_i > 0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v.shape[1])   # np.abs返回絕對值；(Y_v_pred_i<0.5)直接判断小于0.5则true判为1
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i > 0.5)-Y_v_i))).sum(axis=1) > 0).sum()*1.0/Y_v_i.shape[0]  # .sum(axis=1)是把矩阵每一行的数都相加 .shape[0]即行数。0表示第一维行，1表示第二维列
    return ber_test, fer_test


def errors(y_true, y_pred):  # 增加了round函数，有点像误码率了
    # y_pred = 1.0 / (1.0 + np.exp(-1.0 * y_pred))
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))


# def errors(y_true, y_pred):
#   return 1.0*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=-y_pred))


# keras模型定义网络
# 16bit {128,64,32,16,8}
# 32bit {256,128,64,32,16}
model = Sequential()
model.add(Dense(256, activation='relu', use_bias=True, input_dim=code_n))
model.add(BatchNormalization())  # 每层的输入要做标准化
model.add(Dense(256, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(code_n, activation='sigmoid'))  # 模型搭建完用compile来编译模型
optimizer = keras.optimizers.adam(lr=LEARNING_RATE, clipnorm=1.0)  # 如果不设置的话 默认值为 lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[errors])  # 这个error函数到底怎么定义还需要进一步考虑

print('网络参数配置为')
print(model.summary())  # 打印输出检查一下网络

# #################################  Train  ##################################

# 为训练做准备
start_time = time.time()  # 记录训练开始时间
validation_numbers = round(num_of_batch / batch_in_epoch)
BER_all = np.zeros([1, validation_numbers], dtype=np.float32)
validation_numbers = np.arange(validation_numbers).reshape(1, -1)  # 变成向量
# print(BER_all.shape)
# print(validation_numbers.shape)


# 开始训练与测试
for i in range(num_of_batch):  # range是个for循环一样的东西；num_of_batch = 10000

    # training
    training_data, training_labels = create_mix_epoch(code_k, code_n, scaling_factor, is_zeros_word=train_on_zero_word)  # 生成训练数据集，用全0的数据集做训练

    cost = model.train_on_batch(training_data, training_labels)  # 输入的数据就是一组batch，这一组全部算完后更新一次参数

    # validation
    if i % batch_in_epoch == 0:  # batch_in_epoch=400

        print('----------------------------Finish Epoch - ', i / batch_in_epoch, '-----------------------------------')
        # print('训练模型的cost值为：', cost)
        y_validation = np.zeros([1, code_n], dtype=np.float32)
        y_validation_pred = np.zeros([1, code_n], dtype=np.float32)

        for k_sf in validation_snr:  # 测试多个信噪比
            # 测试时格外产生一些数据；用非0的数据集做测试
            validation_data, validation_labels = create_mix_epoch_validation(code_k, code_n, batch_size_validation, k_sf, is_zeros_word=test_on_zero_word)
            ber_val, fer_val = calc_ber_fer(k_sf, y_validation_pred, validation_labels, batch_size_validation * batches_for_val)
            epoch_turns = int(i / batch_in_epoch)
            BER_all[0, epoch_turns] = ber_val

            # 每个信噪比下 误码率与训练次数的关系分别打印，画图，保存到txt
            plt.plot(validation_numbers, BER_all, 'ro')

            # print & write to file
            print('SNR[dB] validation - ', k_sf)
            print('BER validation - ', ber_val)
            print('FER validation - ', fer_val)  # FER frame error rates 误帧率

            # 把每次误码率写入文件
            print('epoch次数： ', epoch_turns, '训练次数：', i, '测试误码率: ', ber_val, '误帧率: ', fer_val, '\n', file=f)

# 记录训练结束时间
end_time = time.time()
print('DNN模型训练次数 ', num_of_batch, '总共花费时间 ', str((end_time - start_time) / 60), ' 分钟 ', file=f)
f.close

##############################################################################################
#  在整个for循环结束，完成全部训练之后：才开始进行画图和存储训练网络这些后续工作
##############################################################################################


# #############################################全部训练完存储模型
model.save('DNN_model_JY.h5')   # 保存模型结构，权重参数，损失函数，优化器，，，所有可以自己配置的东西
model.save_weights('DNN_model_weights_JY.h5')   # 只保留权重参数


# 画图 训练次数影响误码率

plt.grid(True)
legend = []
plt.legend(legend, loc='best')  # 图位置
# plt.axis('tight')  # 不知道是啥
# 图的坐标轴不支持中文显示！！！！蛋疼
plt.xlabel('epoch')
plt.ylabel('BER')
plt.title('BER of train set with epoch increase')
plt.show()


