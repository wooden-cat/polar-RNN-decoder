# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:59:07 2018

@author: bid317
"""

from __future__ import print_function, division
import os  # os模块包含普遍系统的功能，有很多个GPU，指定在第二块GPU上运行本程序
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['HOMEPATH']


import tensorflow as tf
import numpy as np  # NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库
import datetime
from shutil import copyfile
import matplotlib.pyplot as pltf
import sys
import math   # 支持一些数学函数，以及特定的数学变量


code_k = 32     # 信息位码长
code_n = 64   # 总的码长，可以看出来码率0.5
code_rate = 1.0*code_k/code_n   # 算码率，有一个浮点数，最后结果就是浮点数了
n = np.log2(code_n).astype(int)  # BP网络的层数是log2码长
dd = float("30")   # 迭代次数
word_seed = 786000
noise_seed = 345000
start_snr = 1
stop_snr = 1
snr = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)  # np.arange()函数返回一个有终点和起点的固定步长的排列
scaling_factor = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# ########### Neural network config####################

input_output_layer_size = code_n  # 最外面的输入输出层上的神经元数目就是码长
# m=24000
ACTIVITION = tf.nn.sigmoid
N_LAYERS = n   # 总共7层隐藏层
N_HIDDEN_UNITS = code_n # 每层包含N个神经元
# numOfWordSim_train = 20
T_iteration = 10*(n-1)  # 迭代十次的BP算法，等效的网络层数是10*（n-1）
ii = T_iteration
epochnum = 1  # epoch：中文翻译为时期,即所有训练样本的一个正向传递和一个反向传递；一般情况下数据量太大，没法同时通过网络，所以将数据分为几个batch
batch = 1
batch_size = epochnum*batch   # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！！
# 每个batch_size是120个数，分了四个batch，即每次训练的是480个数？每个数训练30个来回
gpu_mem_fraction = 0.99 
clip_tanh = 10.0
batch_in_epoch = 4000  # 每训练400次有一波操作
batches_for_val = 200
# num_of_batch = 10000   # 取名有些混乱，这个是训练的次数
num_of_batch = 10000000   # 取名有些混乱，这个是训练的次数
learning_rate = 0.001
train_on_zero_word = False
test_on_zero_word = False
load_weights = False
is_training = True
HIDDEN_SIZE = 64      # LSTM中隐藏节点的个数。或许也就是每一层有多少个节点？
NUM_LAYERS = 2      # LSTM的层数。
wordRandom = np.random.RandomState(word_seed)  # 伪随机数产生器，（seed）其中seed的值相同则产生的随机数相同
random = np.random.RandomState(noise_seed)


LR_quan=8
LRrel_quan=-4
###########  init parameters ############
W_input1 = np.ones((1, code_n), dtype=np.float32)  # 训练的就是这个系数的值
#Data Generation


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
            i=i+2*n
        n=2*n  # 步长n逐层翻倍
    return x


# 真想吐槽一下，这个代码可读性好差！取名混乱，结构也很乱！烦！！！困！！！！！！
def create_mix_epoch(code_k, code_n, numOfWordSim, scaling_factor, is_zeros_word):  # 把之前的几个函数做集成，开始做整套的编码过程
    X = np.zeros([1,code_n], dtype=np.float32)
    Y = np.zeros([1,code_n], dtype=np.int64)
    
    x = np.zeros([numOfWordSim, code_n],dtype=np.int64)  # numOfWordSim这个玩意代入的参数是batch_size=120
    u = np.zeros([numOfWordSim, code_n],dtype=np.int64)
    
    for sf_i in scaling_factor:
        A = polar_design_awgn(code_n, code_k, sf_i)   # A是bool型的玩意，来判断这个信道是不是合适传输的
        if is_zeros_word:
            d = 0*wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
        else:
            d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))
        # X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
        u[:,A] = d   # u = np.zeros([numOfWordSim, code_n],dtype=np.int64) ，没毛病，u就是120*64的维度，d是120*64的随机数，0,1的随机数，A是64的bool型
        for i in range(0,numOfWordSim):
            x[i] = polar_transform_iter(u[i])

        snr_lin = 10.0**(sf_i/10.0)
        noise = np.sqrt(1.0/(2.0*snr_lin*code_rate))
        X_p_i = random.normal(0.0,1.0,x.shape)*noise + (1)*(1-2*x)  # random.normal按照正态分布取随机数
        x_llr_i = 2*X_p_i/(noise**2)
        X = np.vstack((X,x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
        Y = np.vstack((Y,u))  # u是单纯的原始码

    X = X[1:]  # X是编码加噪声后接收端处理过的对数似然信息
    Y = Y[1:]  # Y是最初未编码的0,1信息
        
    return X, Y


# 计算误码率误帧率
def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    for i in range(0,snr_db.shape[0]):
        A = polar_design_awgn(code_n, code_k, snr_db[i])
        Y_v_pred_i = Y_v_pred[i*numOfWordSim:(i+1)*numOfWordSim,A]
        Y_v_i = Y_v[i*numOfWordSim:(i+1)*numOfWordSim,A]
        ber_test[i] = 2.0*np.abs(((Y_v_pred_i<0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v.shape[1])   # np.abs返回絕對值；(Y_v_pred_i<0.5)直接判断小于0.5则true判为1
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i<0.5)-Y_v_i))).sum(axis=1)>0).sum()*1.0/Y_v_i.shape[0]  # .sum(axis=1)是把矩阵每一行的数都相加 .shape[0]即行数。0表示第一维行，1表示第二维列
    return ber_test, fer_test


with tf.name_scope('inputs'):
    # batch_size=120,code_n即现在的码长64；但batch_size有乘2，因为把左信息和右信息拼接到一起了
    xs = tf.placeholder(tf.float32, shape=[batch_size, code_n], name='x_input')   # xs是编码加噪声后接收端处理过的对数似然信息
    ys = tf.placeholder(tf.float32, shape=[batch_size, code_n], name='y_input')   # ys是最初未编码的0,1信息
    keep_prob = tf.placeholder(tf.float32)  # 占位符，相当于定义了函数参数，但是还不赋值，等到要用了再赋值

#  定义网络
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell')
        for _ in range(NUM_LAYERS)])

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    xs_expand = tf.expand_dims(xs, axis=2)  # 强行拓展到三维，反正不这么干会出错   # 这里有问题！！！三维是干嘛？！！！
    outputs, _ = tf.nn.dynamic_rnn(cell, xs_expand, dtype=tf.float32)
    output = outputs[:, -1, :]  # outputs维度是[batch_size,time,HIDDEN_SIZE],但我们只需要最后时刻的输出
    print(output.shape)

    # 出麻烦了，一增加全连接层就维度变化了，矩阵变成了向量  o((@﹏@))o  难过之后解决这个
    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均平方差损失函数。
    predictions = tf.contrib.layers.fully_connected(output, code_n, activation_fn=None)  # 不设置激活函数，否则默认为ReLU
    # predictions = tf.nn.dropout(predictions, keep_prob)  # 防止过度拟合
    print(predictions.shape)

    # cross entropy loss
    loss = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=predictions))  # tf.reduce_mean取均值
    # 计算损失函数。
    # loss = tf.losses.mean_squared_error(labels=ys, predictions=predictions)  # labels是应该输出的东西，是个参照物；predictions是真正输出的东西；
    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

# #################################  Train  ####################################
# 初始化训练模型
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # 以会话的形式运行运行，sess拥有并管理tensorflow运行时的资源
sess.run(tf.global_variables_initializer())  # 好像缺少sess.close过程
saver = tf.train.Saver()  # 程序要保存或者恢复训练好的模型就要这样
# 开始训练与测试
for i in range(num_of_batch):  # range是个for循环一样的东西；num_of_batch = 10000

    # training
    training_data, training_labels = create_mix_epoch(code_k, code_n, epochnum, scaling_factor, is_zeros_word=train_on_zero_word)  # 生成训练数据集，用全0的数据集做训练
    # 每运行一次更新一次fetch里的值 ; 反正就是在更新网络；没必要用fetch语句；y_output, loss没必要fetch输出
    # 首先占位符申请空间；使用的时候，通过占位符“喂（feed_dict）”给程序。feed_dict的作用是给使用placeholder创建出来的tensor赋值;运行之后用fetch把想要的值给取出来
    _, _, _ = sess.run(fetches=[predictions, loss, train_op], feed_dict={xs: training_data, ys: training_labels, keep_prob: 1})  # 多了个train_step这就是在修改网络参数

    # validation
    if i % batch_in_epoch == 0:  # batch_in_epoch=400
        print('Finish Epoch - ', i/batch_in_epoch)
        y_validation = np.zeros([1,code_n], dtype=np.float32)
        y_validation_pred = np.zeros([1,code_n], dtype=np.float32)
        loss_v = np.zeros([1, 1], dtype=np.float32)

        for k_sf in scaling_factor:   # 测试四个信噪比
            for j in range(batches_for_val):  # batches_for_val=200 每个信噪比下测试的数据集为200个mini-batch

                validation_data, validation_labels = create_mix_epoch(code_k, code_n, batch_size, [k_sf], is_zeros_word=test_on_zero_word)  # 测试时格外产生一些数据；用非0的数据集做测试
                y_validation_pred_j, loss_v_j = sess.run(fetches=[predictions, loss], feed_dict={xs: validation_data, ys: validation_labels, keep_prob: 1})

                y_validation = np.vstack((y_validation,validation_labels))
                y_validation_pred = np.vstack((y_validation_pred,y_validation_pred_j))
                loss_v = np.vstack((loss_v, loss_v_j))

        #JY:公式（11），y_v_pred是公式中x
        y_validation_pred = 1.0 / (1.0 + np.exp(-1.0 * y_validation_pred))   # 用sigmoid函数把输出量化到0~1之间
        # [1:, :]表示第1行开始的保留，第0行去掉。因为第0行是初始化的全0行，并没有任何信息。
        ber_val, fer_val = calc_ber_fer(snr, y_validation_pred[1:, :], y_validation[1:, :], batch_size*batches_for_val)

        # print & write to file
        print('SNR[dB] validation - ', snr)
        print('BER validation - ', ber_val)
        print('FER validation - ', fer_val)  # FER frame error rates 误帧率
