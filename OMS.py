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
#N = 64
#k=8
code_n = 64   # 总的码长，可以看出来码率0.5
code_rate = 1.0*code_k/code_n   # 算码率，有一个浮点数，最后结果就是浮点数了
n = np.log2(code_n).astype(int)  # BP网络的层数是log2码长
dd = float("30")   # 迭代次数
word_seed = 786000
noise_seed = 345000
start_snr = 1
stop_snr = 4
snr = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)  # np.arange()函数返回一个有终点和起点的固定步长的排列
scaling_factor = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)  # 和snr一样一样呢

############ Neural network config####################

input_output_layer_size = code_n  # 最外面的输入输出层上的神经元数目就是码长
#m=24000
ACTIVITION = tf.nn.sigmoid
N_LAYERS = n # 总共7层隐藏层
N_HIDDEN_UNITS = code_n # 每层包含N个神经元
#numOfWordSim_train = 20
T_iteration = 10*(n-1)
ii = T_iteration
epochnum = 30
batch = 4
batch_size = 30*batch
gpu_mem_fraction = 0.99
clip_tanh = 10.0
batch_in_epoch = 400
batches_for_val = 200
num_of_batch = 10000
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

 
#W_input=np.eye(code_n, dtype=np.float32)
#W_input = np.zeros((1, code_n), dtype=np.float32)
W_input1 = np.ones((1, code_n), dtype=np.float32)
#Data Generation


def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0,len(x)):
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)
    return x


def polar_design_awgn(N, k, snr_dB):

    S = 10**(snr_dB/10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
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

#def polar_transform_iter(u): #encoding
#
#    N = len(u)
#    x = np.copy(u)
#    n = np.log2(N).astype(int)
#    for i in range(0,n):
#        C = 2**i
#        B = 2**(n-i)
#        for j in range(C):
#            k = j*B
#            for m in range(0,int(B/2)):
#                ide =int(k+B/2+m)
#                x[k+m] = x[k+m] ^ x[ide]
#                
#    return x
    
def polar_transform_iter(u): #encoding

    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):
        i = 0
        while i < N:
            for j in range(0,n):
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]
            i=i+2*n
        n=2*n
    return x
 
def linear_quantize(v, w_bits, bits_per_delta):
    bound = math.pow(2.0, w_bits)
    min_val, max_val = - bound, bound
    delta = math.pow(2.0, bits_per_delta)
    quantized = np.round(v / delta)
    clipped = np.clip(quantized, min_val, max_val)
    return clipped*delta
    
def create_mix_epoch(code_k,code_n,numOfWordSim,scaling_factor,is_zeros_word):
    X = np.zeros([1,code_n], dtype=np.float32)
    Y = np.zeros([1,code_n], dtype=np.int64)
    
    x = np.zeros([numOfWordSim, code_n],dtype=np.int64)
    u = np.zeros([numOfWordSim, code_n],dtype=np.int64)
    R = np.zeros([1, code_n],dtype=np.float32)
    
    for sf_i in scaling_factor:
        R_mes = dd*np.ones([numOfWordSim, code_n],dtype=np.float32)
        A = polar_design_awgn(code_n, code_k, sf_i)
        if is_zeros_word:
            d = 0*wordRandom.randint(0, 2, size=(numOfWordSim, code_k))
        else:
            d = wordRandom.randint(0, 2, size=(numOfWordSim, code_k))
        
        u[:,A] = d
        R_mes[:,A] = 0
        
        for i in range(0,numOfWordSim):
            x[i] = polar_transform_iter(u[i])


        snr_lin = 10.0**(sf_i/10.0)
        noise = np.sqrt(1.0/(2.0*snr_lin*code_rate))
        X_p_i = random.normal(0.0,1.0,x.shape)*noise + (1)*(1-2*x)
        x_llr_i = 2*X_p_i/(noise**2)
        # X_i = 1/(1+np.exp(x_llr_r_i))
        R = np.vstack((R,R_mes))
        X = np.vstack((X,x_llr_i))
        Y = np.vstack((Y,u))
        
        ber_ini = np.abs(((x_llr_i<0)-x)).sum()/(x_llr_i.shape[0]*x_llr_i.shape[1])
        #print(ber_ini)
    # X = X[1:] - np.repeat(np.mean(X[1:],axis=1)[:,np.newaxis], X[1:].shape[1], axis=1)
    # X = X/np.repeat(np.sqrt(np.var(X,axis=1))[:,np.newaxis], X.shape[1], axis=1)
    R = R[1:]
    X = X[1:]
    X = np.vstack((X,R))
    Y = Y[1:]
        
    return X,Y

def calc_ber_fer(snr_db, Y_v_pred, Y_v, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    for i in range(0,snr_db.shape[0]):
        A = polar_design_awgn(code_n, code_k, snr_db[i])
        Y_v_pred_i = Y_v_pred[i*numOfWordSim:(i+1)*numOfWordSim,A]
        Y_v_i = Y_v[i*numOfWordSim:(i+1)*numOfWordSim,A]
        ber_test[i] = 2.0*np.abs(((Y_v_pred_i<0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v.shape[1])
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i<0.5)-Y_v_i))).sum(axis=1)>0).sum()*1.0/Y_v_i.shape[0]

    return ber_test, fer_test

R_mes_zeros = np.zeros([batch_size, int(code_n/2)], dtype=np.float32)
R_mes_input = np.zeros([batch_size, int(code_n)], dtype=np.float32)



keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('inputs'):  # 让几个变量有相同的名字，啥意思？
    xs = tf.placeholder(tf.float32, shape=[2*batch_size, code_n], name='x_input')
    ys = tf.placeholder(tf.float32, shape=[batch_size, code_n], name='y_input')
#d = np.random.randint(0,2,size=(m,k), dtype=bool)

# first layer
net_dict = {}

 

for i in range(0,T_iteration,1):
    net_dict["Scale_weight_{0}".format(i)] = tf.Variable(W_input1)
    #net_dict["Offset_weight_{0}".format(i)] = tf.Variable(W_input)
    
    #net_dict["Scale_weight_{0}".format(i)] = quantize(net_dict["Scale_weight_{0}".format(i)],k,None)
    #net_dict["hidden_weight_{0}".format(i)] = tf.multiply(W_input, net_dict["hidden_weight_{0}".format(i)])
    if (int((int(i/(n-1)))%2)==0):
        if (int(i%(n-1))==0):                                              
            net_dict["hidden_left_x_0{0}".format(i)] = tf.slice(xs, [batch_size, 0], [batch_size, code_n])
        else:
            net_dict["hidden_left_x_0{0}".format(i)] = net_dict["hidden_left_x_{0}".format(i-1)]
                                                      
        if (i<n):
            net_dict["hidden_left_x_1{0}".format(i)] = R_mes_input
        else:
            net_dict["hidden_left_x_1{0}".format(i)] = net_dict["hidden_right_x_{0}".format(int((n-1)*(int(i/(n-1)))-i%(n-1)-1))]                                          
                                                      
        net_dict["hidden_left_x_2{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_0{0}".format(i)],[batch_size*2**(int(n-1-i%(n-1))),-1])
        net_dict["hidden_left_x_3{0}".format(i)] = tf.slice(net_dict["hidden_left_x_2{0}".format(i)],[0,0],[batch_size*2**(int(n-1-i%(n-1))),2**int((i%(n-1)))])
        net_dict["hidden_left_x_4{0}".format(i)] = tf.tile(net_dict["hidden_left_x_3{0}".format(i)],multiples=[1,2])
        net_dict["hidden_left_x_5{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_4{0}".format(i)],[batch_size,int(code_n)])#R_i,j
        
         
        net_dict["hidden_left_x_6{0}".format(i)] = tf.reshape(net_dict["hidden_left_x_1{0}".format(i)],[batch_size*2**(int(n-1-i%(n-1))),-1])
        net_dict["hidden_left_x_7{0}".format(i)] = tf.slice(net_dict["hidden_left_x_6{0}".format(i)],[0,int(2**(i%(n-1)))],[batch_size*2**(int(n-1-i%(n-1))),2**int((i%(n-1)))])
        net_dict["hidden_left_x_8{0}".format(i)] = tf.zeros(shape=[batch_size*2**(int(n-1-i%(n-1))),int(2**(i%(n-1)))],dtype=tf.float32)
        net_dict["hidden_left_x_9{0}".format(i)] = tf.slice(net_dict["hidden_left_x_6{0}".format(i)],[0,0],[batch_size*2**(int(n-1-i%(n-1))),int(2**(i%(n-1)))])
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
        #net_dict["hidden_left_x_21{0}".format(i)] = tf.nn.relu(tf.subtract(net_dict["hidden_left_x_20{0}".format(i)],net_dict["Offset_weight_{0}".format(i)]))
        
        net_dict["hidden_left_x_21{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_19{0}".format(i)],net_dict["hidden_left_x_20{0}".format(i)])
        net_dict["hidden_left_x_22{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_21{0}".format(i)],net_dict["Scale_weight_{0}".format(i)])
        #net_dict["hidden_left_x_22{0}".format(i)] = tf.multiply(net_dict["hidden_left_x_21{0}".format(i)],net_dict["hidden_weight_{0}".format(i)])
        #net_dict["hidden_left_x_22{0}".format(i)] = tf.nn.relu(net_dict["hidden_left_x_22{0}".format(i)])
        #  net_dict["hidden_left_x_22{0}".format(i)] = tf.clip_by_value(net_dict["hidden_left_x_22{0}".format(i)],clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        net_dict["hidden_left_x_{0}".format(i)] = tf.add(net_dict["hidden_left_x_22{0}".format(i)],net_dict["hidden_left_x_14{0}".format(i)])

                                                      
    else:
        if (int(i%(n-1))==0):
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
with tf.name_scope("Weight_lastlayer"):        
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
#net_dict["hidden_right_x_21{0}".format(ii)] = tf.nn.relu(tf.subtract(net_dict["hidden_right_x_20{0}".format(ii)],net_dict["Offset_weight_{0}".format(ii)]))
net_dict["hidden_right_x_21{0}".format(ii)] = tf.multiply(net_dict["hidden_right_x_19{0}".format(ii)],net_dict["hidden_right_x_20{0}".format(ii)])
net_dict["hidden_right_x_22{0}".format(ii)] = tf.multiply(net_dict["hidden_right_x_21{0}".format(ii)],net_dict["Scale_weight_{0}".format(ii)])
#net_dict["hidden_right_x_22{0}".format(ii)] = tf.clip_by_value(net_dict["hidden_right_x_22{0}".format(ii)],clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
net_dict["hidden_right_x_{0}".format(ii)] = tf.add(net_dict["hidden_right_x_22{0}".format(ii)],net_dict["hidden_right_x_14{0}".format(ii)])

with tf.name_scope("y_output"):
    #y_output = tf.add(net_dict["hidden_right_x_{0}".format(ii)],net_dict["hidden_right_x_1{0}".format(ii)])
    y_output = net_dict["hidden_right_x_{0}".format(ii)]
    y_output = tf.nn.dropout(y_output, keep_prob)
with tf.name_scope("loss"):

    # cross entropy loss
    loss = 1.0*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys,logits=-y_output))
    
    tf.summary.scalar('loss',loss)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
#train_op_norm, cost_norm, layers_inputs_norm = built_net(ls, rs, ys, norm=True)

grad_dict = {}
visual_grad_dict = {}


##################################  Train  ####################################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
#merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter("logs/", sess.graph)
saver = tf.train.Saver()
#f_results = open(results_path, 'w+')
train_loss_vec = np.zeros(1,dtype=np.float32)
val_loss_vec = np.zeros(1,dtype=np.float32)
#print(sess.run(net_dict["hidden_weight_{0}".format(0)]))
#print(sess.run(net_dict["hidden_weight_{0}".format(1)]))

# load model
if load_weights:
    saver.restore(sess, weights_path)

for i in range(num_of_batch):

    # training
    training_data, training_labels = create_mix_epoch(code_k,code_n,epochnum,scaling_factor,is_zeros_word=train_on_zero_word)
     
    training_labels_for_mse = training_labels
    #training_data = linear_quantize(training_data, LR_quan, LRrel_quan)

    y_train, train_loss, _ = sess.run(fetches=[y_output, loss, train_step], feed_dict={xs: training_data, ys: training_labels_for_mse, keep_prob: 1})
    #print('train_loss',train_loss)
    #for j in range(T_iteration+1):   
     #   weight = sess.run(net_dict["Scale_weight_{0}".format(j)])
        #weight1 = sess.run(net_dict["Offset_weight_{0}".format(j)])
        #net_dict["Scale_weight_{0}".format(j)].load(linear_quantize(weight, int_quan, Del_quan), sess)
        #net_dict["Offset_weight_{0}".format(j)].load(linear_quantize(weight1, int_quan, Del_quan), sess)
  
    if(i%batch_in_epoch == 0):
       
        
        # result = sess.run(merged, feed_dict={xs: training_data, ys: training_labels_for_mse})
       # writer.add_summary(result, i)
        #print(sess.run(net_dict["Offset_weight_{0}".format(50)]))
        #print(sess.run(net_dict["Scale_weight_{0}".format(50)]))
        print('Finish Epoch - ', i/batch_in_epoch)

        # validation
        y_v = np.zeros([1,code_n], dtype=np.float32)
        y_v_pred = np.zeros([1,code_n], dtype=np.float32)
        loss_v = np.zeros([1, 1], dtype=np.float32)

        for k_sf in scaling_factor:
            for j in range(batches_for_val):

                x_v_j, y_v_j = create_mix_epoch(code_k,code_n,batch_size,[k_sf],is_zeros_word=test_on_zero_word)
                #create_mix_epoch([k_sf], wordRandom, batch_size, code_n, code_k, code_generatorMatrix, is_zeros_word=test_on_zero_word)
                y_v_pred_j, loss_v_j = sess.run(fetches = [y_output, loss] ,feed_dict={xs:x_v_j, ys:y_v_j,keep_prob: 1})
                
                y_v = np.vstack((y_v,y_v_j))
                y_v_pred = np.vstack((y_v_pred,y_v_pred_j))
                loss_v = np.vstack((loss_v, loss_v_j))

        #JY:公式（11），y_v_pred是公式中x
        y_v_pred = 1.0 / (1.0 + np.exp(-1.0 * y_v_pred))

        ber_val, fer_val = calc_ber_fer(snr, y_v_pred[1:,:], y_v[1:,:], batch_size*batches_for_val)

        # print & write to file
        print('SNR[dB] validation - ', snr)
        print('BER validation - ', ber_val)
        print('FER validation - ', fer_val)
