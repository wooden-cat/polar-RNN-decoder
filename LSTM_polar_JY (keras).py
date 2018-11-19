# -*- coding: utf-8 -*-
"""
训练数据是固定码长下全部可能码字都进行训练
测试时随机产生码子进行测试
目前的毛病就是训练速度太慢了一些
训练几个小时，最终0.03误码率左右
"""

from __future__ import print_function, division

import tensorflow as tf
import numpy as np  # NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库
import keras
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda
from keras.optimizers import RMSprop, Adam
from keras.layers import SimpleRNN,LSTM, GRU, Activation
from keras.layers.wrappers import  Bidirectional
from keras import backend as K

import datetime
from shutil import copyfile
import matplotlib.pyplot as pltf
import sys
import math   # 支持一些数学函数，以及特定的数学变量

code_k = 8     # 信息位码长
code_n = 16   # 总的码长，可以看出来码率0.5
code_rate = 1.0*code_k/code_n   # 算码率，有一个浮点数，最后结果就是浮点数了
word_seed = 786000
noise_seed = 345000
start_snr = 6
stop_snr = 6
snr = np.arange(start_snr, stop_snr+1, 1, dtype=np.float32)  # np.arange()函数返回一个有终点和起点的固定步长的排列
scaling_factor = np.arange(start_snr, stop_snr+1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# ########### Neural network config####################
# epoch：中文翻译为时期,即所有训练样本的一个正向传递和一个反向传递；一般情况下数据量太大，没法同时通过网络，所以将数据分为几个batch
epochnum = 256   # 每次训练这么多组code_n bit的码字，必须为2**code_k
batch = 1
batch_size = epochnum*batch   # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！
batch_size_validation = 16
batch_in_epoch = 100    # 每训练400次有一波操作
batches_for_val = 5     # 貌似使用这个来计算误帧率,要有多个帧才能计算误帧率
num_of_batch = 80000   # 取名有些混乱，这个是训练的次数
LEARNING_RATE = 0.0003  # 学习率 不设置的话函数自动默认是0.001
train_on_zero_word = False
test_on_zero_word = False
load_weights = False
is_training = True
HIDDEN_SIZE = 64     # LSTM中隐藏层，就是状态向量的维度
wordRandom = np.random.RandomState(word_seed)  # 伪随机数产生器，（seed）其中seed的值相同则产生的随机数相同
random = np.random.RandomState(noise_seed)


def bitrevorder(x):
    m = np.amax(x)  # 输入的数组里最大的数
    n = np.ceil(np.log2(m)).astype(int)  # 这个最大的数用二进制表示有n位
    for i in range(0,len(x)):  # i从0到len(x),这个序列有i位，都要反转
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)  # int将字符串转为整形，这里是转为2进制整形。[::-1]序列倒序输出
    return x                          # str.format 把format里的东西放在str对应的位置，例如："Hello, {0} and {1}!".format("John", "Mary")


def polar_design_awgn(N, k, snr_dB):
    S = 10**(snr_dB/10)  # 计算信噪比公式，10log10S/N，反着来，得到信息值
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)  # 高斯密度进化，选出合格的信道
    for j in range(1,int(np.log2(N))+1):
        u = 2**j
        for t in range(0,int(u/2)):
            T = z0[t]
            z0[t] = 2*T - T**2     # upper channel
            z0[int(u/2)+t] = T**2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))

    A = np.zeros(N, dtype=bool)
    A[idx] = True

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
    increment = np.hstack((np.zeros(k-1, dtype=int), np.ones(1, dtype=int)))
    a = add_bool(a, increment)
    return a

    
def polar_transform_iter(u): #encoding
    N = len(u)  # 返回对象长度
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):  # 编码做log2N层的运算，每层都把这N个元素处理个遍
        i = 0
        while i < N:  # i是N个元素中的第i个
            for j in range(0,n): # 每轮的步长是n，j表示n步长内的操作
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]  # ^按位异或
            i = i+2*n
        n = 2*n  # 步长n逐层翻倍
    return x


#Data Generation
def create_mix_epoch(code_k, code_n, numOfWordSim, scaling_factor, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1,code_n], dtype=np.float32)
    Y = np.zeros([1,code_k], dtype=np.int64)
    
    x = np.zeros([numOfWordSim, code_n], dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    u = np.zeros([numOfWordSim, code_n], dtype=np.int64)
    d = np.zeros([numOfWordSim, code_k], dtype=np.int64)
    for sf_i in scaling_factor:
        A = polar_design_awgn(code_n, code_k, sf_i)   # A是bool型的玩意，来判断这个信道是不是合适传输的
        # print("A是这个东西", A)
        # #### 在这里加入循环！！！！！！！！！！！！！！
        if is_zeros_word:  # 用全0数据训练
            d = 0*wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
        else:
            # 把d变成1，2,3,4,5然后转化为2进制，从而遍历所有的情况，看看是不是我的网络设置有毛病
            for k in range(1, numOfWordSim):   # 在码长固定的情况下遍历所有的可能情况
                d[k] = inc_bin(d[k - 1])
            # d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # 随机生成训练数据

        # print(d)
        # X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
        u[:, A] = d   # u = np.zeros([numOfWordSim, code_n],dtype=np.int64) ，没毛病，u就是120*64的维度，d是120*64的随机数，0,1的随机数，A是64的bool型
        for i in range(0,numOfWordSim):
            x[i] = polar_transform_iter(u[i])

        snr_lin = 10.0**(sf_i/10.0)
        noise = np.sqrt(1.0/(2.0*snr_lin*code_rate))
        X_p_i = random.normal(0.0,1.0,x.shape)*noise + (1)*(1-2*x)  # random.normal按照正态分布取随机数
        x_llr_i = 2*X_p_i/(noise**2)
        X = np.vstack((X,x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
        Y = np.vstack((Y,d))  # u是单纯的原始码

    X = X[1:]  # X是编码加噪声后接收端处理过的对数似然信息
    Y = Y[1:]  # Y是最初未编码的0,1信息

    return X, Y


def create_mix_epoch_validation(code_k, code_n, numOfWordSim, scaling_factor, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1, code_n], dtype=np.float32)
    Y = np.zeros([1, code_k], dtype=np.int64)

    x = np.zeros([numOfWordSim, code_n], dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    u = np.zeros([numOfWordSim, code_n], dtype=np.int64)
    d = np.zeros([numOfWordSim, code_k], dtype=np.int64)
    for sf_i in scaling_factor:
        A = polar_design_awgn(code_n, code_k, sf_i)  # A是bool型的玩意，来判断这个信道是不是合适传输的
        # print("A是这个东西", A)
        # #### 在这里加入循环！！！！！！！！！！！！！！
        if is_zeros_word:  # 用全0数据训练
            d = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
        else:
            # 把d变成1，2,3,4,5然后转化为2进制，从而遍历所有的情况，看看是不是我的网络设置有毛病
            # for k in range(1, numOfWordSim):  # 在码长固定的情况下遍历所有的可能情况
            #   d[k] = inc_bin(d[k - 1])
            d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # 随机生成训练数据

        # print(d)
        # X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
        u[:, A] = d  # u = np.zeros([numOfWordSim, code_n],dtype=np.int64) ，没毛病，u就是120*64的维度，d是120*64的随机数，0,1的随机数，A是64的bool型
        for i in range(0, numOfWordSim):
            x[i] = polar_transform_iter(u[i])

        snr_lin = 10.0 ** (sf_i / 10.0)
        noise = np.sqrt(1.0 / (2.0 * snr_lin * code_rate))
        X_p_i = random.normal(0.0, 1.0, x.shape) * noise + (1) * (1 - 2 * x)  # random.normal按照正态分布取随机数
        x_llr_i = 2 * X_p_i / (noise ** 2)
        X = np.vstack((X, x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
        Y = np.vstack((Y, d))  # u是单纯的原始码

    X = X[1:]  # X是编码加噪声后接收端处理过的对数似然信息
    Y = Y[1:]  # Y是最初未编码的0,1信息

    return X, Y

# 计算误码率误帧率
def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    # print('Y_v_pred.shape', Y_v_pred.shape)
    # print('Y_v.shape', Y_v.shape)
    for i in range(0,snr_db.shape[0]):
        Y_v_pred_i = Y_v_pred[i*numOfWordSim:(i+1)*numOfWordSim]
        # print('Y_v_pred_i.shape', Y_v_pred_i.shape)
        Y_v_i = Y_v[i*numOfWordSim:(i+1)*numOfWordSim]
        # 实在把不准这个误码的判决条件！这个还需要仔细考虑！！！
        ber_test[i] = np.abs(((Y_v_pred_i > 0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v.shape[1])   # np.abs返回絕對值；(Y_v_pred_i<0.5)直接判断小于0.5则true判为1
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i > 0.5)-Y_v_i))).sum(axis=1) > 0).sum()*1.0/Y_v_i.shape[0]  # .sum(axis=1)是把矩阵每一行的数都相加 .shape[0]即行数。0表示第一维行，1表示第二维列
    return ber_test, fer_test


def errors(y_true, y_pred):  # 增加了round函数，有点像误码率了
   # y_pred = 1.0 / (1.0 + np.exp(-1.0 * y_pred))
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

# def errors(y_true, y_pred):
#   return 1.0*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=-y_pred))

'''
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, shape=[batch_size, code_n], name='x_input')   # xs是编码加噪声后接收端处理过的对数似然信息
    ys = tf.placeholder(tf.float32, shape=[batch_size, code_k], name='y_input')   # ys是最初未编码的0,1信息
    keep_prob = tf.placeholder(tf.float32)  # 占位符，相当于定义了函数参数，但是还不赋值，等到要用了再赋值
'''
# keras模型定义网络
model = Sequential()
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, recurrent_dropout=0.5),  # 双向LSTM
                            input_shape=(None, 1)))
model.add(BatchNormalization())  # 每层的输入要做标准化
model.add(Bidirectional(LSTM(HIDDEN_SIZE, recurrent_dropout=0.5, )))
model.add(BatchNormalization())
model.add(Dense(code_k, activation='sigmoid'))  # 模型搭建完用compile来编译模型
optimizer = keras.optimizers.adam(lr=LEARNING_RATE, clipnorm=1.0)  # 如果不设置的话 默认值为 lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[errors])  # 这个error函数到底怎么定义还需要进一步考虑
print(model.summary())   # 打印输出检查一下网络
# #################################  Train  ##################################
# 开始训练与测试
for i in range(num_of_batch):  # range是个for循环一样的东西；num_of_batch = 10000

    # training
    training_data, training_labels = create_mix_epoch(code_k, code_n, epochnum, scaling_factor, is_zeros_word = train_on_zero_word)  # 生成训练数据集，用全0的数据集做训练
    # print(training_labels.shape)
    # print(training_data.shape)
    training_data = tf.reshape(training_data, (-1, 16, 1))
    # training_labels = tf.reshape(training_labels, (-1, 8, 1))
    # print(training_labels.shape)
    # print(training_data.shape)
    cost = model.train_on_batch(training_data, training_labels)   # 感觉这句有问题，或许改成fit会更好？ 输入的数据就是一组batch，这一组batch一起更新一次参数

    # validation
    if i % batch_in_epoch == 0:  # batch_in_epoch=400
        print('Finish Epoch - ', i/batch_in_epoch)
        print('训练模型的cost值为：', cost)
        y_validation = np.zeros([1,code_k], dtype=np.float32)
        y_validation_pred = np.zeros([1,code_k], dtype=np.float32)

        for k_sf in scaling_factor:   # 测试四个信噪比
            for j in range(batches_for_val):  # 为了让最终测试的误码率更可靠，计算batches_for_val组数据。最后算平均误码率。

                validation_data, validation_labels = create_mix_epoch_validation(code_k, code_n, batch_size_validation, [k_sf], is_zeros_word=test_on_zero_word)  # 测试时格外产生一些数据；用非0的数据集做测试
                # print(validation_data.shape)
                validation_data = tf.reshape(validation_data, (-1, 16, 1))
                # print(validation_data.shape)
                y_validation_pred_j = model.predict(validation_data, steps=1)  # 这里的输出是个范围很大的数，不是局限在0~1之间的
                # print("预测值y_validation_pred_j形状是：", y_validation_pred_j.shape)
                # print('y_validation_pred_j', y_validation_pred_j)

                y_validation = np.vstack((y_validation, validation_labels))  # 用于验证的发送端产生的原始数据
                y_validation_pred = np.vstack((y_validation_pred, y_validation_pred_j))
        # print('y_validation.shape', y_validation.shape)
        # print('y_validation_pred.shape', y_validation_pred.shape)
        # y_validation_pred = 1.0 / (1.0 + np.exp(-1.0 * y_validation_pred))   # 用sigmoid函数把输出量化到0~1之间
        ber_val, fer_val = calc_ber_fer(snr, y_validation_pred[1:, :], y_validation[1:, :], batch_size_validation*batches_for_val)

        # print & write to file
        print('SNR[dB] validation - ', snr)
        print('BER validation - ', ber_val)
        print('FER validation - ', fer_val)  # FER frame error rates 误帧率


# 在整个for循环结束，完成全部训练之后：才开始进行画图和存储训练网络这些后续工作

# 全部训练完存储模型
model.save('LSTM_model_JY.h5')   # 保存模型结构，权重参数，损失函数，优化器，，，所有可以自己配置的东西
model.save_weights('LSTM_model_weights_JY.h5')   # 只保留权重参数
