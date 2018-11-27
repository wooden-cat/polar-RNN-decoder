from __future__ import print_function, division

import tensorflow as tf
import numpy as np  # NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库
from keras.models import load_model
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
code_k = 16 # 信息位码长
code_n = 32  # 总的码长，可以看出来码率0.5
code_rate = 1.0 * code_k / code_n  # 算码率，有一个浮点数，最后结果就是浮点数了
size_nn = 8        # 每次分的小块的大小
word_seed = 786000
noise_seed = 345000

# 训练信噪比
start_snr = 6
stop_snr = 6
scaling_factor = np.arange(start_snr, stop_snr + 1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# 测试信噪比序列
validation_start_snr = 6
validation_stop_snr = 6
validation_snr = np.arange(validation_start_snr, validation_stop_snr + 1, 1,
                           dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组
batch_size_validation = 1  # 用于验证的码字有这么多一组
SNR_validation = 6

times_of_validation = 10    # 反复测试，以减少随机性
# ########### Neural network config####################
# epoch：中文翻译为时期,即所有训练样本的一个正向传递和一个反向传递；一般情况下数据量太大，没法同时通过网络，所以将数据分为几个batch
epochnum = 2 ** code_k  # 每次训练这么多组code_n bit的码字，必须为2**code_k
batch = 1
batch_size = epochnum * batch  # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！
batch_in_epoch = 100  # 每训练这么多次有一波计算误码率的操作
batches_for_val = 1  # 貌似使用这个来计算误帧率,要有多个帧才能计算误帧率
num_of_batch = 5000  # 取名有些混乱，这个是训练的次数
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
    return A


def polar_design_awgn_AA(N, k, snr_dB):
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
    AA = np.zeros(N, dtype=int)
    AA[idx] = 1
    return AA


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


def create_mix_epoch_validation(code_k, code_n, numOfWordSim, validation_snr, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1, code_n], dtype=np.float32)
    Y = np.zeros([1, code_n], dtype=np.int64)

    x = np.zeros([numOfWordSim, code_n], dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    u = np.zeros([numOfWordSim, code_n], dtype=np.int64)
    d = np.zeros([numOfWordSim, code_k], dtype=np.int64)
    for sf_i in validation_snr:
        A = polar_design_awgn(code_n, code_k, sf_i)  # A是bool型的玩意，来判断这个信道是不是合适传输的
        # print("A是这个东西", A)
        # #### 在这里加入循环！！！！！！！！！！！！！！
        if is_zeros_word:  # 用全0数据训练
            d = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
        else:
            d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # 随机生成训练数据

        u[:,A] = d
        for i in range(0, numOfWordSim):
            x[i] = polar_transform_iter(u[i])

        snr_lin = 10.0 ** (sf_i / 10.0)
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


def errors(y_true, y_pred):  # 增加了round函数，有点像误码率了
    # y_pred = 1.0 / (1.0 + np.exp(-1.0 * y_pred))
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))


def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    for i in range(0, snr_db.shape[0]):
        A = polar_design_awgn(code_n, code_k, snr_db[i])
        Y_v_pred_i = Y_v_pred[i*numOfWordSim:(i+1)*numOfWordSim,A]
        Y_v_i = Y_v[i*numOfWordSim:(i+1)*numOfWordSim,A]
        ber_test[i] = 2.0*np.abs(((Y_v_pred_i > 0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v.shape[1])   # np.abs返回絕對值；(Y_v_pred_i<0.5)直接判断小于0.5则true判为1
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i > 0.5)-Y_v_i))).sum(axis=1)>0).sum()*1.0/Y_v_i.shape[0]  # .sum(axis=1)是把矩阵每一行的数都相加 .shape[0]即行数。0表示第一维行，1表示第二维列
    return ber_test, fer_test


def matrix_and(a, b):
    a = int(a)
    b = int(b)
    for i in range(0, len(a)):
        a[i] = a[i] & b[i]
    return a
# ##################################################
# 开始译码
# ##################################################


ber_test = np.zeros(times_of_validation, dtype=np.float32)

'''
# 直接相减，对差值取绝对值
for i in range(times_of_validation):
    y_llr, validation_X = create_mix_epoch_validation(code_k, code_n, batch_size_validation, validation_snr, is_zeros_word=test_on_zero_word)

    model = load_model('DNN_model_JY.h5', custom_objects={'errors': errors})
    y_llr1_matrix = y_llr.reshape(-1, size_nn)  # 一行16个做好分组
    print(y_llr1_matrix)
    partion_num = 4
    X = np.zeros([4, size_nn])
    # 倒数第一块
    X[3, :] = model.predict((y_llr1_matrix[3, :]).reshape(-1, size_nn), steps=1)
    # X[3, :] = model.predict(np.zeros([1, 16]), steps=1)
    X1_llr = polar_transform_iter(X[3, :] > 0.5)
    # print(X1_llr)
    # ###########################################
    # 这里不是简单的相减的运算！！！！！！！
    # 原运算是位与，与运算的反运算！！！！
    # 倒数第二块
    X[2, :] = model.predict((np.abs(y_llr1_matrix[2, :] - X1_llr)).reshape(-1, size_nn), steps=1)
    X2_llr = polar_transform_iter(X[[2, 3], :].reshape(1, -1) > 0.5)
    X2_llr = X2_llr.reshape(-1, size_nn)
    # 倒数三四块
    y_llr2_matrix = y_llr1_matrix[[2, 3], :] - X2_llr
    # 倒数三块
    X[partion_num-3, :] = model.predict(np.abs(y_llr2_matrix[1, :]).reshape(-1, size_nn), steps=1)
    X3_llr = polar_transform_iter(X[partion_num-3, :] > 0.5)
    # 倒数第四块
    X[partion_num-4, :] = model.predict((np.abs(y_llr2_matrix[0, :] - X3_llr)).reshape(-1, size_nn), steps=1)
    # 译码完成
    X = X.reshape(1, -1)

    # 计算误码率
    ber_test[0, i] = np.abs(((X > 0.5)-validation_X)).sum()/(validation_X.shape[0]*validation_X.shape[1])

    # print(y_llr1_matrix)
    # print((X > 0.5).reshape(-1, 8))
    # print(validation_X.reshape(-1, 8))
    # print((X > 0.5).reshape(-1, 8) - validation_X.reshape(-1, 8))
    u = np.zeros([1, 32], dtype=np.int64)
    A = polar_design_awgn(code_n, code_k, SNR_validation)
    A = A.reshape(1, 32)
    for i in range(32):
        if A[0, i]:
            u[0, i] = ((X > 0.5) - validation_X)[0, i]
    BER_information = np.abs(u).sum()/32
    print('单纯信息位的误码率是', BER_information)
'''
A = polar_design_awgn_AA(code_n, code_k, SNR_validation)
A = A.reshape(-1, size_nn)


for i in range(times_of_validation):
    # 前期准备
    X = np.zeros([4, size_nn], dtype=np.int64)
    y_llr, validation_X = create_mix_epoch_validation(code_k, code_n, batch_size_validation, validation_snr, is_zeros_word=test_on_zero_word)
    model = load_model('DNN_model_JY.h5', custom_objects={'errors': errors})
    y_llr1_matrix = y_llr.reshape(-1, size_nn)  # 一行16个做好分组
    # print(y_llr1_matrix)
    # print(np.rint(y_llr1_matrix))
    partion_num = 4

    # 倒数第一块
    X[3, :] = model.predict((y_llr1_matrix[3, :]).reshape(-1, size_nn), steps=1) + 0.5
    for k in range(size_nn):
        X[3, k] = X[3, k] & A[3, k]
    X1_llr = polar_transform_iter(X[3, :])
    # print(X1_llr)

    # 倒数第二块
    NN2_input = np.abs(y_llr1_matrix[2, :] - X1_llr).reshape(-1, size_nn)
    X[2, :] = model.predict(NN2_input, steps=1) + 0.5
    for k in range(size_nn):
        X[2, k] = X[2, k] & A[2, k]

    X2_llr = polar_transform_iter(X[[2, 3], :].reshape(1, -1))
    X2_llr = X2_llr.reshape(-1, size_nn)

    # 倒数三四块
    NN34_input = y_llr1_matrix[[2, 3], :] - X2_llr
    # 倒数三块
    NN3_input = np.abs(NN34_input[1, :]).reshape(-1, size_nn)
    X[1, :] = model.predict(NN3_input, steps=1) + 0.5
    for k in range(size_nn):
        X[1, k] = X[1, k] & A[1, k]
    X3_llr = polar_transform_iter(X[1, :])

    # 倒数第四块
    NN4_input = (np.abs(NN34_input[0, :] - X3_llr)).reshape(-1, size_nn)
    X[0, :] = model.predict(NN4_input, steps=1) + 0.5
    for k in range(size_nn):
        X[0, k] = X[0, k] & A[0, k]
    # 译码完成
    X = X.reshape(1, -1)

    # 计算误码率
    # print(validation_X.shape)
    ber_test[i] = np.abs(X-validation_X).sum()/(validation_X.shape[0]*validation_X.shape[1])
    print(i, '次译码测试出错的码字矩阵')
    print((X-validation_X).reshape(-1, 8))
    print(A)

    # print(y_llr1_matrix)
    # print((X > 0.5).reshape(-1, 8))
    # print(validation_X.reshape(-1, 8))
    # print((X > 0.5).reshape(-1, 8) - validation_X.reshape(-1, 8))
    '''
    u = np.zeros([1, 32], dtype=np.int64)
    A = polar_design_awgn(code_n, code_k, SNR_validation)
    A = A.reshape(1, 32)
    for i in range(32):
        if A[0, i]:
            u[0, i] = ((X > 0.5) - validation_X)[0, i]
    BER_information = np.abs(u).sum()/32
    print('单纯信息位的误码率是', BER_information)
    '''

print('信息位和冻结位混合的误码率是', np.mean(ber_test))
