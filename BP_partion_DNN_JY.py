# -*- coding: utf-8 -*-
"""
by wooden_cat
BP分块训练，每个小块是8bit，其他部分就是常规的BP，到倒数第三层的时候变成DNN的神经网络。
译码到还剩下4层的时候停止。
码长暂定为128，有16个DNN小块。
"""

from __future__ import print_function, division
# import os  # os模块包含普遍系统的功能，有很多个GPU，指定在第二块GPU上运行本程序，我的傻逼电脑没有两个GPU呀
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['HOMEPATH']
 
import tensorflow as tf
import numpy as np  # NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库
import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import sys
import math   # 支持一些数学函数，以及特定的数学变量
import time


code_k = 64     # 信息位码长
code_n = 128   # 总的码长，可以看出来码率0.5
DNN_size = 8  # 定义分块神经网络小块的大小
code_rate = 1.0*code_k/code_n   # 算码率，有一个浮点数，最后结果就是浮点数了
n = np.log2(code_n).astype(int)  # BP网络的层数是log2码长
dd = float("30")   # 迭代次数
word_seed = 786000
noise_seed = 345000
start_snr = 1
stop_snr = 4
snr = np.arange(start_snr, stop_snr+1, 1, dtype=np.float32)  # np.arange()函数返回一个有终点和起点的固定步长的排列
scaling_factor = np.arange(start_snr, stop_snr+1, 1, dtype=np.float32)  # arrang返回一个数组，也就是始末信噪比的数组

# ########### Neural network config####################

input_output_layer_size = code_n  # 最外面的输入输出层上的神经元数目就是码长
# m=24000
ACTIVITION = tf.nn.sigmoid
N_LAYERS = n   # 总共7层隐藏层
N_HIDDEN_UNITS = code_n # 每层包含N个神经元
# numOfWordSim_train = 20
T_iteration = 10*(n-1)  # 迭代十次的BP算法，等效的网络层数是10*（n-1）
ii = T_iteration
epochnum = 30  # epoch：中文翻译为时期,即所有训练样本的一个正向传递和一个反向传递；一般情况下数据量太大，没法同时通过网络，所以将数据分为几个batch
batch = 4
batch_size = 30*batch   # batch_size是指将多个数据同时作为输入  ！！！非常重要的一个变量！！！
# 每个batch_size是120个数，分了四个batch，即每次训练的是480个数？每个数训练30个来回
gpu_mem_fraction = 0.99 
clip_tanh = 10.0
batch_in_epoch = 400  # 每训练400次有一波操作
batches_for_val = 200
num_of_batch = 10000   # 取名有些混乱，这个是训练的次数
learning_rate = 0.001
train_on_zero_word = True
test_on_zero_word = False
load_weights = False
wordRandom = np.random.RandomState(word_seed)  # 伪随机数产生器，（seed）其中seed的值相同则产生的随机数相同
random = np.random.RandomState(noise_seed)
int_quan=8
Del_quan=-7

LR_quan=8
LRrel_quan=-4
###########  init parameters ############
W_input1 = np.ones((1, code_n), dtype=np.float32)  # 训练的就是这个系数的值
#Data Generation


def bitrevorder(x):
    m = np.amax(x) # 输入的数组里最大的数
    n = np.ceil(np.log2(m)).astype(int)  # 这个最大的数用二进制表示有n位
    for i in range(0,len(x)): # i从0到len(x),这个序列有i位，都要反转
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2) # int将字符串转为整形，这里是转为2进制整形。[::-1]序列倒序输出
    return x                          # str.format 把format里的东西放在str对应的位置，例如："Hello, {0} and {1}!".format("John", "Mary")

def polar_design_awgn(N, k, snr_dB):

    S = 10**(snr_dB/10) # 计算信噪比公式，10log10S/N，反着来，得到信息值
    z0 = np.zeros(N)

    z0[0] = np.exp(-S) # 高斯密度进化，选出合格的信道
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
    A_matrix = A.reshape(-1, 8)
    print('8个分为一组冻结位有多少个', A_matrix)
    A_zeros = np.zeros([A_matrix.shape[0], A_matrix.shape[1]], dtype=np.float32)
    A_count = (A_matrix - A_zeros).sum(axis=1)
    print('每一行的信息位有多少个', A_count)
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
    R = np.zeros([1, code_n],dtype=np.float32)
    
    for sf_i in scaling_factor:
        R_mes = dd*np.ones([numOfWordSim, code_n],dtype=np.float32)
        A = polar_design_awgn(code_n, code_k, sf_i)   # A是bool型的玩意，来判断这个信道是不是合适传输的
        if is_zeros_word:
            d = 0*wordRandom.randint(0, 2, size=(numOfWordSim, code_k))  # max取值只能到2，不能到1
        else:
            d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))
        # X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
        u[:,A] = d   # u = np.zeros([numOfWordSim, code_n],dtype=np.int64) ，没毛病，u就是120*64的维度，d是120*64的随机数，0,1的随机数，A是64的bool型
        R_mes[:,A] = 0   # 右信息的初始化，用A判断是不是信息位。信息位取0，冻结位是无穷大
        
        for i in range(0,numOfWordSim):
            x[i] = polar_transform_iter(u[i])

        snr_lin = 10.0**(sf_i/10.0)
        noise = np.sqrt(1.0/(2.0*snr_lin*code_rate))
        X_p_i = random.normal(0.0,1.0,x.shape)*noise + (1)*(1-2*x)  # random.normal按照正态分布取随机数
        x_llr_i = 2*X_p_i/(noise**2)
        # X_i = 1/(1+np.exp(x_llr_r_i))
        R = np.vstack((R,R_mes))  # np.vstack沿着竖直方向将矩阵堆叠起来。
        X = np.vstack((X,x_llr_i))  # x_llr_i是接收端用来译码的对数似然信息
        Y = np.vstack((Y,u))  # u是单纯的原始码

    R = R[1:]
    X = X[1:]  # slice操作，0位丢掉，把1位和之后位保留，搞毛？这样的话为什么FOR循环里不少循环一遍？
    X = np.vstack((X,R))  # 所以x最后变成上下两块，上面一块左信息，下面一块右信息
    Y = Y[1:]
        
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


R_mes_zeros = np.zeros([batch_size, int(code_n/2)], dtype=np.float32)  # 这个变量似乎定义了没有用到
R_mes_input = np.zeros([batch_size, int(code_n)], dtype=np.float32)

keep_prob = tf.placeholder(tf.float32)  # tensorflow中的占位符，相当于定义了函数参数，但是还不赋值，等到要用了再赋值
with tf.name_scope('inputs'):
    # batch_size=120,code_n即现在的码长64；但batch_size有乘2，因为把左信息和右信息拼接到一起了
    xs = tf.placeholder(tf.float32, shape=[2*batch_size, code_n], name='x_input')   # xs是对数似然之后的接收端信息
    ys = tf.placeholder(tf.float32, shape=[batch_size, code_n], name='y_input')    # ys是最初未编码的0,1信息

# first layer
net_dict = {}  # 搞毛？不懂

for i in range(0, T_iteration, 1):   # T_iteration为10*(n-1)是神经网络的层数，从左到右做初始化
    net_dict["Scale_weight_{0}".format(i)] = tf.Variable(W_input1)  # Scale_weight_{0}这个变量一层有64个和神经网络一样有10*（n-1）层

    if int((int(i/(n-1))) % 2) == 0:    # 每层迭代的网络有n层，但不要最左边一层，所以n-1是视觉上的一层。就是（n-1）*2层完成向左又向右的一轮。找出所有的最右排，赋值为似然比。
        if int(i % (n-1)) == 0:            # %为返回除法的余数
            net_dict["hidden_left_x_0{0}".format(i)] = tf.slice(xs, [batch_size, 0], [batch_size, code_n])   # 字符串中大括号内的数字分别对应着format的几个参数  这个就是最右排的初始化为对数似然比
        else:
            net_dict["hidden_left_x_0{0}".format(i)] = net_dict["hidden_left_x_{0}".format(i-1)]
                                                      
        if i < n:
            # str.format(),应用在这句话中即给str赋值得到了hidden_left_x1{i}这个string；用dict字典形式记录string的值为R_mes_input,不过这个R_mes_input像是个矩阵呀
            net_dict["hidden_left_x_1{0}".format(i)] = R_mes_input
        else:
            net_dict["hidden_left_x_1{0}".format(i)] = net_dict["hidden_right_x_{0}".format(int((n-1)*(int(i/(n-1)))-i%(n-1)-1))]
        # 这里的-1是缺省值，把tensor整成batch_size*2**(int(n-1-i%(n-1)))的形状，能够弄几个就几个。若不是-1，eg是2，则表示弄2个这种形状。
        net_dict["hidden_left_x_2{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_0{0}".format(i)],[batch_size*2**(int(n-1-i%(n-1))),-1])
        # tf.slice(input_, begin, size, name = None)
        net_dict["hidden_left_x_3{0}".format(i)] = tf.slice(net_dict["hidden_left_x_2{0}".format(i)],[0,0],[batch_size*2**(int(n-1-i%(n-1))),2**int((i%(n-1)))])
        # tile(input,multiples）像铺瓷砖一样，把原输入重复输入multiples次
        net_dict["hidden_left_x_4{0}".format(i)] = tf.tile(net_dict["hidden_left_x_3{0}".format(i)],multiples=[1,2])
        net_dict["hidden_left_x_5{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_4{0}".format(i)],[batch_size,int(code_n)])#R_i,j

        net_dict["hidden_left_x_6{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_1{0}".format(i)],[batch_size*2**(int(n-1-i%(n-1))),-1])
        net_dict["hidden_left_x_7{0}".format(i)] = tf.slice(net_dict["hidden_left_x_6{0}".format(i)],[0,int(2**(i%(n-1)))],[batch_size*2**(int(n-1-i%(n-1))),2**int((i%(n-1)))])
        # batch_size*2**(int(n-1-i%(n-1)))和2**(i%(n-1)两个玩意反复出现好多次，我觉得另设置变量来表示应该会更好
        net_dict["hidden_left_x_8{0}".format(i)] = tf.zeros(shape=[batch_size*2**(int(n-1-i%(n-1))),int(2**(i%(n-1)))],dtype=tf.float32)
        net_dict["hidden_left_x_9{0}".format(i)] = tf.slice(net_dict["hidden_left_x_6{0}".format(i)],[0,0],[batch_size*2**(int(n-1-i%(n-1))),int(2**(i%(n-1)))])
        # contact就是连起来；0是第一个维度，1是第二个维度
        net_dict["hidden_left_x_10{0}".format(i)] = tf.concat([net_dict["hidden_left_x_7{0}".format(i)], net_dict["hidden_left_x_9{0}".format(i)]],1)     
        net_dict["hidden_left_x_11{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_10{0}".format(i)],[batch_size,int(code_n)])             #[L_i+n/2^j, L_i]
        
        net_dict["hidden_left_x_12{0}".format(i)] = tf.slice(net_dict["hidden_left_x_2{0}".format(i)],[0,int(2**(i%(n-1)))],[batch_size*2**(int(n-1-i%(n-1))),int(2**(i%(n-1)))])
        net_dict["hidden_left_x_13{0}".format(i)] = tf.concat([net_dict["hidden_left_x_8{0}".format(i)],net_dict["hidden_left_x_12{0}".format(i)]],1)   #[0,R_i+n/2^j]
        net_dict["hidden_left_x_14{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_13{0}".format(i)],[batch_size,int(code_n)])

        net_dict["hidden_left_x_15{0}".format(i)] = tf.concat([net_dict["hidden_left_x_12{0}".format(i)],net_dict["hidden_left_x_8{0}".format(i)]],1)    #[R_i+n/2^j,0]
        net_dict["hidden_left_x_15{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_15{0}".format(i)],[batch_size,int(code_n)])
        
        net_dict["hidden_left_x_16{0}".format(i)] = tf.sign(net_dict["hidden_left_x_5{0}".format(i)])    #sign(R_i,j)
        net_dict["hidden_left_x_17{0}".format(i)] = tf.add(net_dict["hidden_left_x_11{0}".format(i)],net_dict["hidden_left_x_15{0}".format(i)])
        net_dict["hidden_left_x_18{0}".format(i)] = tf.sign(net_dict["hidden_left_x_17{0}".format(i)])  #sign
        net_dict["hidden_left_x_19{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_16{0}".format(i)],net_dict["hidden_left_x_18{0}".format(i)])
        net_dict["hidden_left_x_20{0}".format(i)] = tf.where(tf.greater(tf.abs(net_dict["hidden_left_x_5{0}".format(i)]),tf.abs(net_dict["hidden_left_x_17{0}".format(i)])),
                                                               tf.abs(net_dict["hidden_left_x_17{0}".format(i)]),
                                                               tf.abs(net_dict["hidden_left_x_5{0}".format(i)]))
        
        net_dict["hidden_left_x_21{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_19{0}".format(i)],net_dict["hidden_left_x_20{0}".format(i)])
        net_dict["hidden_left_x_22{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_21{0}".format(i)],net_dict["Scale_weight_{0}".format(i)])

        net_dict["hidden_left_x_{0}".format(i)] = tf.add(net_dict["hidden_left_x_22{0}".format(i)],net_dict["hidden_left_x_14{0}".format(i)])

    else:
        if int(i % (n-1) == 0):
            net_dict["hidden_right_x_0{0}".format(i)] = tf.slice(xs, [0, 0], [batch_size, code_n])
        else:
            net_dict["hidden_right_x_0{0}".format(i)] = net_dict["hidden_right_x_{0}".format(i-1)]
         
        net_dict["hidden_right_x_1{0}".format(i)] = net_dict["hidden_left_x_{0}".format(int((n-1)*(int(i/(n-1)))-i%(n-1)-1))]

        net_dict["hidden_right_x_2{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_0{0}".format(i)],[batch_size*2**(int(i%(n-1))),-1])
        net_dict["hidden_right_x_3{0}".format(i)] = tf.slice(net_dict["hidden_right_x_2{0}".format(i)],[0,0],[batch_size*2**(int(i%(n-1))),int(2**(n-i%(n-1)-1))])
        net_dict["hidden_right_x_4{0}".format(i)] = tf.tile(net_dict["hidden_right_x_3{0}".format(i)],multiples=[1,2])
        net_dict["hidden_right_x_5{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_4{0}".format(i)],[batch_size,int(code_n)])      #L_i,j

        net_dict["hidden_right_x_6{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_1{0}".format(i)],[batch_size*2**(int(i%(n-1))),-1])
        net_dict["hidden_right_x_7{0}".format(i)] = tf.slice(net_dict["hidden_right_x_6{0}".format(i)],[0,2**int((n-i%(n-1)-1))],[batch_size*2**(int(i%(n-1))),2**int((n-i%(n-1)-1))])
        net_dict["hidden_right_x_8{0}".format(i)] = tf.zeros(shape=[batch_size*2**(int(i%(n-1))),2**int((n-i%(n-1)-1))],dtype=tf.float32)
        net_dict["hidden_right_x_9{0}".format(i)] = tf.slice(net_dict["hidden_right_x_6{0}".format(i)],[0,0],[batch_size*2**(int(i%(n-1))),2**int((n-i%(n-1)-1))])
        net_dict["hidden_right_x_10{0}".format(i)] = tf.concat([net_dict["hidden_right_x_7{0}".format(i)], net_dict["hidden_right_x_9{0}".format(i)]],1)
        net_dict["hidden_right_x_11{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_10{0}".format(i)],[batch_size,int(code_n)])     #[R_i+n/2^j, R_i]
        
        net_dict["hidden_right_x_12{0}".format(i)] = tf.slice(net_dict["hidden_right_x_2{0}".format(i)],[0,2**int((n-i%(n-1)-1))],[batch_size*2**(int(i%(n-1))),2**int((n-i%(n-1)-1))])
        net_dict["hidden_right_x_13{0}".format(i)] = tf.concat([net_dict["hidden_right_x_8{0}".format(i)],net_dict["hidden_right_x_12{0}".format(i)]],1)   #[0,R_i+n/2^j]
        net_dict["hidden_right_x_14{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_13{0}".format(i)],[batch_size,int(code_n)])#[0,L_i+n/2^j]

        net_dict["hidden_right_x_15{0}".format(i)] = tf.concat([net_dict["hidden_right_x_12{0}".format(i)],net_dict["hidden_right_x_8{0}".format(i)]],1)    #[R_i+n/2^j,0]
        net_dict["hidden_right_x_15{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_15{0}".format(i)],[batch_size,int(code_n)])    #[L_i+n/2^j,0]

        net_dict["hidden_right_x_16{0}".format(i)] = tf.sign(net_dict["hidden_right_x_5{0}".format(i)])    #sign(L_i,j)
        net_dict["hidden_right_x_17{0}".format(i)] = tf.add(net_dict["hidden_right_x_11{0}".format(i)],net_dict["hidden_right_x_15{0}".format(i)])
        net_dict["hidden_right_x_18{0}".format(i)] = tf.sign(net_dict["hidden_right_x_17{0}".format(i)])  #sign
        net_dict["hidden_right_x_19{0}".format(i)] = tf.multiply(net_dict["hidden_right_x_16{0}".format(i)],net_dict["hidden_right_x_18{0}".format(i)])
        net_dict["hidden_right_x_20{0}".format(i)] = tf.where(tf.greater(tf.abs(net_dict["hidden_right_x_5{0}".format(i)]),tf.abs(net_dict["hidden_right_x_17{0}".format(i)])),
                                                               tf.abs(net_dict["hidden_right_x_17{0}".format(i)]),
                                                               tf.abs(net_dict["hidden_right_x_5{0}".format(i)]))
        #net_dict["hidden_right_x_21{0}".format(i)] = tf.nn.relu(tf.subtract(net_dict["hidden_right_x_20{0}".format(i)],net_dict["Offset_weight_{0}".format(i)]))
        net_dict["hidden_right_x_21{0}".format(i)] = tf.multiply(net_dict["hidden_right_x_19{0}".format(i)],net_dict["hidden_right_x_20{0}".format(i)])
        net_dict["hidden_right_x_22{0}".format(i)] = tf.multiply(net_dict["hidden_right_x_21{0}".format(i)],net_dict["Scale_weight_{0}".format(i)])
        #  net_dict["hidden_right_x_22{0}".format(i)] = tf.clip_by_value(net_dict["hidden_right_x_22{0}".format(i)],clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        net_dict["hidden_right_x_{0}".format(i)] = tf.add(net_dict["hidden_right_x_22{0}".format(i)],net_dict["hidden_right_x_14{0}".format(i)])

        #####the last hidden layer######
with tf.name_scope("Weight_lastlayer"):  # 定义了一个名字作用域，后面变量的名字都带这个开头，默认变成了Weight_lastlayer/hidden_right_x_0{0}
    net_dict["Scale_weight_{0}".format(ii)] = tf.Variable(W_input1)
    #net_dict["Offset_weight_{0}".format(ii)] = tf.Variable(W_input)
    #tf.summary.histogram('T_iteration' + '/weights', net_dict["Offset_weight_{0}".format(ii)])
    #net_dict["hidden_weight_{0}".format(ii)] = tf.multiply(W_input, net_dict["hidden_weight_{0}".format(ii)])
    net_dict["hidden_right_x_0{0}".format(ii)] = net_dict["hidden_right_x_{0}".format(T_iteration-1)]
    net_dict["hidden_right_x_1{0}".format(ii)] = tf.slice(xs, [batch_size, 0], [batch_size, code_n])

    net_dict["hidden_right_x_2{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_0{0}".format(ii)],[batch_size*2**(n-1),-1])
    net_dict["hidden_right_x_3{0}".format(ii)] = tf.slice(net_dict["hidden_right_x_2{0}".format(ii)],[0,0],[batch_size*2**(n-1),1])
    net_dict["hidden_right_x_4{0}".format(ii)] = tf.tile(net_dict["hidden_right_x_3{0}".format(ii)],multiples=[1,2])
    net_dict["hidden_right_x_5{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_4{0}".format(ii)],[batch_size,int(code_n)])     #L_i,j

    net_dict["hidden_right_x_6{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_1{0}".format(ii)],[batch_size*2**(n-1),-1])
    net_dict["hidden_right_x_7{0}".format(ii)] = tf.slice(net_dict["hidden_right_x_6{0}".format(ii)],[0,1],[batch_size*2**(n-1),1])
    net_dict["hidden_right_x_8{0}".format(ii)] = tf.zeros(shape=[batch_size*2**(n-1),1],dtype=tf.float32)
    net_dict["hidden_right_x_9{0}".format(ii)] = tf.slice(net_dict["hidden_right_x_6{0}".format(ii)],[0,0],[batch_size*2**(n-1),1])
    net_dict["hidden_right_x_10{0}".format(ii)] = tf.concat([net_dict["hidden_right_x_7{0}".format(ii)], net_dict["hidden_right_x_9{0}".format(ii)]],1)
    net_dict["hidden_right_x_11{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_10{0}".format(ii)],[batch_size,int(code_n)])     #[R_i+n/2^j, R_i]

    net_dict["hidden_right_x_12{0}".format(ii)] = tf.slice(net_dict["hidden_right_x_2{0}".format(ii)],[0,1],[batch_size*2**(n-1),1])
    net_dict["hidden_right_x_13{0}".format(ii)] = tf.concat([net_dict["hidden_right_x_8{0}".format(ii)],net_dict["hidden_right_x_12{0}".format(ii)]],1)   #[0,R_i+n/2^j]
    net_dict["hidden_right_x_14{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_13{0}".format(ii)],[batch_size,int(code_n)])#[0,L_i+n/2^j]

    net_dict["hidden_right_x_15{0}".format(ii)] = tf.concat([net_dict["hidden_right_x_12{0}".format(ii)],net_dict["hidden_right_x_8{0}".format(ii)]],1)    #[R_i+n/2^j,0]
    net_dict["hidden_right_x_15{0}".format(ii)] = tf.reshape(net_dict["hidden_right_x_15{0}".format(ii)],[batch_size,int(code_n)])    #[L_i+n/2^j,0]

    net_dict["hidden_right_x_16{0}".format(ii)] = tf.sign(net_dict["hidden_right_x_5{0}".format(ii)])    #sign(L_i,j)
    net_dict["hidden_right_x_17{0}".format(ii)] = tf.add(net_dict["hidden_right_x_11{0}".format(ii)],net_dict["hidden_right_x_15{0}".format(ii)])
    net_dict["hidden_right_x_18{0}".format(ii)] = tf.sign(net_dict["hidden_right_x_17{0}".format(ii)])  #sign
    net_dict["hidden_right_x_19{0}".format(ii)] = tf.multiply(net_dict["hidden_right_x_16{0}".format(ii)],net_dict["hidden_right_x_18{0}".format(ii)])
    net_dict["hidden_right_x_20{0}".format(ii)] = tf.where(tf.greater(tf.abs(net_dict["hidden_right_x_5{0}".format(ii)]),tf.abs(net_dict["hidden_right_x_17{0}".format(ii)])),
                                                       tf.abs(net_dict["hidden_right_x_17{0}".format(ii)]),
                                                       tf.abs(net_dict["hidden_right_x_5{0}".format(ii)]))
    net_dict["hidden_right_x_21{0}".format(ii)] = tf.multiply(net_dict["hidden_right_x_19{0}".format(ii)],net_dict["hidden_right_x_20{0}".format(ii)])
    net_dict["hidden_right_x_22{0}".format(ii)] = tf.multiply(net_dict["hidden_right_x_21{0}".format(ii)],net_dict["Scale_weight_{0}".format(ii)])
    net_dict["hidden_right_x_{0}".format(ii)] = tf.add(net_dict["hidden_right_x_22{0}".format(ii)],net_dict["hidden_right_x_14{0}".format(ii)])

with tf.name_scope("y_output"):
    #y_output = tf.add(net_dict["hidden_right_x_{0}".format(ii)],net_dict["hidden_right_x_1{0}".format(ii)])
    y_output = net_dict["hidden_right_x_{0}".format(ii)]
    y_output = tf.nn.dropout(y_output, keep_prob)  # 防止过度拟合
with tf.name_scope("loss"):

    # cross entropy loss
    loss = 1.0*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys,logits=-y_output))  # tf.reduce_mean取均值
    # 可视化，画图的。暂时可以不管
    #tf.summary.scalar('loss',loss)
# Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# #################################  Train  ####################################

# 初始化训练模型
start_time = time.time()  # 记录训练开始时间
f = open('OMS_JY.txt', 'w')   # 开个文档 记录测试的结果

BER_all = np.zeros([1, stop_snr - start_snr + 1], dtype=np.float32)
# (BER_all.shape)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction) # 每个GPU中显存使用的上限是gpu_mem_fraction
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # 以会话的形式运行运行，sess拥有并管理tensorflow运行时的资源
sess.run(tf.global_variables_initializer()) # 好像缺少sess.close过程
saver = tf.train.Saver()  # 程序要保存或者恢复训练好的模型就要这样

# load model
if load_weights:  # 加载权重，这是开始用训练好的网络去做测试？ 不用管就好  load_weights = False 不行的东西
    saver.restore(sess, weights_path)

for i in range(num_of_batch):  # range是个for循环一样的东西；num_of_batch = 10000

    # training
    training_data, training_labels = create_mix_epoch(code_k, code_n, epochnum, scaling_factor, is_zeros_word=train_on_zero_word) # 生成训练数据集，用全0的数据集做训练
    training_labels_for_mse = training_labels
    # 每运行一次更新一次fetch里的值 ; 反正就是在更新网络；没必要用fetch语句；y_output, loss没必要fetch输出
    # 首先占位符申请空间；使用的时候，通过占位符“喂（feed_dict）”给程序。feed_dict的作用是给使用placeholder创建出来的tensor赋值;运行之后用fetch把想要的值给取出来
    y_train, train_loss, _ = sess.run(fetches=[y_output, loss, train_step], feed_dict={xs: training_data, ys: training_labels_for_mse, keep_prob: 1})  # 多了个train_step这就是在修改网络参数

    if i % batch_in_epoch == 0:  # batch_in_epoch=400

        print('Finish Epoch - ', i/batch_in_epoch)

        # validation
        y_v = np.zeros([1,code_n], dtype=np.float32)
        y_v_pred = np.zeros([1,code_n], dtype=np.float32)
        loss_v = np.zeros([1, 1], dtype=np.float32)

        for k_sf in scaling_factor:   # 测试四个信噪比
            for j in range(batches_for_val):  # batches_for_val=200 每个信噪比下测试的数据集为200个mini-batch

                x_v_j, y_v_j = create_mix_epoch(code_k,code_n,batch_size,[k_sf],is_zeros_word=test_on_zero_word)  # 测试时格外产生一些数据；用非0的数据集做测试
                y_v_pred_j, loss_v_j = sess.run(fetches=[y_output, loss], feed_dict={xs:x_v_j, ys: y_v_j, keep_prob: 1})
                
                y_v = np.vstack((y_v,y_v_j))
                y_v_pred = np.vstack((y_v_pred,y_v_pred_j))
                loss_v = np.vstack((loss_v, loss_v_j))

        #JY:公式（11），y_v_pred是公式中x
        y_v_pred = 1.0 / (1.0 + np.exp(-1.0 * y_v_pred))   # 用sigmoid函数把输出量化到0~1之间
        # [1:, :]表示第1行开始的保留，第0行去掉。因为第0行是初始化的全0行，并没有任何信息。
        ber_val, fer_val = calc_ber_fer(snr, y_v_pred[1:, :], y_v[1:, :], batch_size*batches_for_val)
        # print(ber_val.shape)
        BER_all = np.vstack((BER_all, ber_val))
        # print & write to file
        print('SNR[dB] validation - ', snr)
        print('BER validation - ', ber_val)
        print('FER validation - ', fer_val)  # FER frame error rates 误帧率
        # 误码率等信息写入文件记录
        print('epoch次数： ', int(i/batch_in_epoch), '训练次数：', i, '测试误码率: ', ber_val, '误帧率: ', fer_val, '\n', file=f)

end_time = time.time()
print('训练次数 ', num_of_batch, '总共花费时间 ', str((end_time-start_time)/60), ' 分钟 ', file=f)
f.close

# ####################################
# 为最终结果画图
BER_all = BER_all[1:]  # 去掉第一行的全0
validation_numbers = round(num_of_batch/batch_in_epoch)  # 一共测试验证多少次
validation_numbers = np.arange(validation_numbers)   # 变成向量
print(validation_numbers)
print(validation_numbers.shape)
print(BER_all)
print(BER_all[:, 0].shape)
plt.plot(validation_numbers, BER_all[:, 0], 'ro')
plt.plot(validation_numbers, BER_all[:, 1], 'bo')
plt.plot(validation_numbers, BER_all[:, 2], 'yo')
plt.plot(validation_numbers, BER_all[:, 3], 'go')
plt.grid(True)
legend = []
plt.yscale('log')
plt.legend(legend, loc='best')  # 图位置
# plt.axis('tight')  # 不知道是啥
# 图的坐标轴不支持中文显示！！！！蛋疼
plt.xlabel('epoch')
plt.ylabel('BER')
plt.title('BER of train set with epoch increase')
plt.show()
