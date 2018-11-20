# -*- coding: utf-8 -*-
"""
改成DNN的网络，也还是keras

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
import time

from shutil import copyfile
import matplotlib.pyplot as plt
import sys
import math   # 支持一些数学函数，以及特定的数学变量

# 极化码编码参数
code_k = 8     # 信息位码长
code_n = 16   # 总的码长，可以看出来码率0.5
code_rate = 1.0*code_k/code_n   # 算码率，有一个浮点数，最后结果就是浮点数了
word_seed = 786000
noise_seed = 345000

# 训练信噪比
start_snr = 6
stop_snr = 6
scaling_factor = np.arange(start_snr, stop_snr+1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# 测试信噪比序列
validation_start_snr = 6
validation_stop_snr = 6
validation_snr = np.arange(validation_start_snr, validation_stop_snr+1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组
batch_size_validation = 16  # 用于验证的码字有这么多一组
# ########### Neural network config####################
# epoch：中文翻译为时期,即所有训练样本的一个正向传递和一个反向传递；一般情况下数据量太大，没法同时通过网络，所以将数据分为几个batch
epochnum = 256   # 每次训练这么多组code_n bit的码字，必须为2**code_k
batch = 1
batch_size = epochnum*batch   # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！
batch_in_epoch = 100    # 每训练这么多次有一波计算误码率的操作
batches_for_val = 10     # 貌似使用这个来计算误帧率,要有多个帧才能计算误帧率
num_of_batch = 10000  # 取名有些混乱，这个是训练的次数
LEARNING_RATE = 0.0001  # 学习率 不设置的话函数自动默认是0.001
train_on_zero_word = False
test_on_zero_word = False
load_weights = False
is_training = True
HIDDEN_SIZE = 64     # LSTM中隐藏节
NUM_LAYERS = 2      # LSTM的层数。
wordRandom = np.random.RandomState(word_seed)  # 伪随机数产生器，（seed）其中seed的值相同则产生的随机数相同
random = np.random.RandomState(noise_seed)


def bitrevorder(x):
    m = np.amax(x)  # 输入的数组里最大的数
    n = np.ceil(np.log2(m)).astype(int)  # 这个最大的数用二进制表示有n位
    for i in range(0, len(x)):  # i从0到len(x),这个序列有i位，都要反转
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


def create_mix_epoch_validation(code_k, code_n, numOfWordSim, validation_snr, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1, code_n], dtype=np.float32)
    Y = np.zeros([1, code_k], dtype=np.int64)

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
f = open('BER_DNN.txt', 'w')  # 开个文件，记录测试的误码率

model = Sequential()
model.add(Dense(128, activation='relu', use_bias=True, input_dim=16))
model.add(BatchNormalization())  # 每层的输入要做标准化
model.add(Dense(64, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu', use_bias=True))
model.add(BatchNormalization())
model.add(Dense(8, activation='sigmoid'))  # 模型搭建完用compile来编译模型
optimizer = keras.optimizers.adam(lr=LEARNING_RATE, clipnorm=1.0)  # 如果不设置的话 默认值为 lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[errors])  # 这个error函数到底怎么定义还需要进一步考虑

print('网络参数配置为')
print(model.summary())   # 打印输出检查一下网络

# #################################  Train  ##################################

# 为训练做准备
start_time = time.time()  # 记录训练开始时间
validation_numbers = round(num_of_batch/batch_in_epoch)
BER_all = np.zeros([1, validation_numbers], dtype=np.float32)
validation_numbers = np.arange(validation_numbers).reshape(1, -1)  # 变成向量
# print(BER_all.shape)
# print(validation_numbers.shape)


# 开始训练与测试
for i in range(num_of_batch):  # range是个for循环一样的东西；num_of_batch = 10000

    # training
    training_data, training_labels = create_mix_epoch(code_k, code_n, epochnum, scaling_factor, is_zeros_word = train_on_zero_word)  # 生成训练数据集，用全0的数据集做训练

    '''
    # 调换顺序
    # 但是我担心这个调换过于频繁，影响运算速度
    arr = np.arange(epochnum)  # K是信息位的长度，把所有可能出现序列的序号都列出
    np.random.shuffle(arr)  # 只对多维矩阵的第一维做打乱处理，改变排列顺序
    training_data = training_data[arr]  # 相当于按照同样的规律把training_data和training_labels按照行调换了顺序
    training_labels = training_labels[arr]
    '''
    cost = model.train_on_batch(training_data, training_labels)   # 感觉这句有问题，或许改成fit会更好？ 输入的数据就是一组batch，这一组batch一起更新一次参数

    # validation
    if i % batch_in_epoch == 0:  # batch_in_epoch=400

        print('----------------------------Finish Epoch - ', i/batch_in_epoch, '-----------------------------------')
        # print('训练模型的cost值为：', cost)
        y_validation = np.zeros([1,code_k], dtype=np.float32)
        y_validation_pred = np.zeros([1,code_k], dtype=np.float32)

        for k_sf in validation_snr:   # 测试多个信噪比
            for j in range(batches_for_val):  # 为了让最终测试的误码率更可靠，每个信噪比下计算batches_for_val组数据。最后算平均误码率以及误帧率。

                validation_data, validation_labels = create_mix_epoch_validation(code_k, code_n, batch_size_validation, [k_sf], is_zeros_word=test_on_zero_word)  # 测试时格外产生一些数据；用非0的数据集做测试
                # print(validation_data.shape)
                # validation_data = tf.reshape(validation_data, (-1, 16, 1))
                # print(validation_data.shape)
                y_validation_pred_j = model.predict(validation_data, steps=1)  # 这里的输出是个范围很大的数，不是局限在0~1之间的
                # print("预测值y_validation_pred_j形状是：", y_validation_pred_j.shape)
                # print('y_validation_pred_j', y_validation_pred_j)

                y_validation = np.vstack((y_validation, validation_labels))  # 用于验证的发送端产生的原始数据
                y_validation_pred = np.vstack((y_validation_pred, y_validation_pred_j))
        # print('y_validation.shape', y_validation.shape)
        # print('y_validation_pred.shape', y_validation_pred.shape)
        # y_validation_pred = 1.0 / (1.0 + np.exp(-1.0 * y_validation_pred))   # 用sigmoid函数把输出量化到0~1之间
        ber_val, fer_val = calc_ber_fer(validation_snr, y_validation_pred[1:, :], y_validation[1:, :], batch_size_validation*batches_for_val)
        BER_all[0, int(i/batch_in_epoch)] = ber_val

        '''
        # print & write to file
        print('SNR[dB] validation - ', validation_snr)
        print('BER validation - ', ber_val)
        print('FER validation - ', fer_val)  # FER frame error rates 误帧率
        '''


        # 把每次误码率写入文件
        print('训练次数：', i, '测试误码率: ', ber_val, '误帧率: ', fer_val, '\n', file=f)


# 记录训练结束时间
end_time = time.time()
print('DNN模型训练次数 ', num_of_batch, '总共花费时间 ', str((end_time-start_time)/60), ' 分钟 ', file=f)
print('\n 训练后的网络保存在 DNN_model_JY.h5 \n 训练后的参数保留在 DNN_model_weights_JY.h5')
f.close

# 在整个for循环结束，完成全部训练之后：才开始进行画图和存储训练网络这些后续工作

# 全部训练完存储模型
model.save('DNN_model_JY.h5')   # 保存模型结构，权重参数，损失函数，优化器，，，所有可以自己配置的东西
model.save_weights('DNN_model_weights_JY.h5')   # 只保留权重参数


# 画图 训练次数影响误码率
plt.plot(validation_numbers, BER_all, 'ro')
plt.grid(True)
legend = []
plt.legend(legend, loc='best')  # 图位置
# plt.axis('tight')  # 不知道是啥
# 图的坐标轴不支持中文显示！！！！蛋疼
plt.xlabel('epoch')
plt.ylabel('BER')
plt.title('BER of train set with epoch increase')
plt.show()

# 画图 不同信噪比下误码率
# 不仅画当前的神经网路，还要画几个对比函数都是事先保存的误码率
# 都是从别人代码抄的二手误码率，未必可靠
'''
# ML = [0.018563,    0.0071536,    0.0021272,   0.00037756,    4.625e-05]
BP = [0.232650402652771,  0.190316205533597,    0.177832650018636,   0.164884291725105,    0.142405545399147,
      0.128362117780294]

plt.plot(validation_snr, BP, '-r')   # BP和DNN的误码率是保存后直接打印画图
legend = []
plt.plot(validation_snr, ber_val)   # 只有RNN的误码率是现场计算的
legend.append('BP')
legend.append('DNN')
# legend.append('RNN')
plt.legend(legend, loc='best')
plt.yscale('log')
plt.xlabel('$GSNR$')
plt.ylabel('BER')
plt.grid(True)
plt.show()
'''


