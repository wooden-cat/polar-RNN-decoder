
"""
在固定信噪比下，判断8bit截断之后，每个小段信息位的数量
"""

import tensorflow as tf
import numpy as np

snr_dB = 5
code_n = 128
code_k = 64
code_rate = 0.5
NN_size = 8
numOfWordSim = 10
VALIDATION_SNR = [4, 5, 6]
train_on_zero_word = True


noise_seed = 345000
word_seed = 786000
n = np.log2(code_n).astype(int)
N_LAYERS = n - np.log2(NN_size).astype(int)  # 一共需要这么多层
random = np.random.RandomState(noise_seed)
wordRandom = np.random.RandomState(word_seed)

def bitrevorder(x):
    m = np.amax(x) # 输入的数组里最大的数
    n = np.ceil(np.log2(m)).astype(int)  # 这个最大的数用二进制表示有n位
    for i in range(0,len(x)): # i从0到len(x),这个序列有i位，都要反转
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2) # int将字符串转为整形，这里是转为2进制整形。[::-1]序列倒序输出
    return x


def polar_design_awgn(N, k, snr_dB):
    S = 10**(snr_dB/10)  # 计算信噪比公式，10log10S/N，反着来，得到信息值
    z0 = np.zeros(N)
    z0[0] = np.exp(-S)  # 高斯密度进化，选出合格的信道
    for j in range(1, int(np.log2(N))+1):
        u = 2**j
        for t in range(0, int(u/2)):
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


def polar_transform_iter(u):
    N = len(u)  # 返回对象长度
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):  # 编码做log2N层的运算，每层都把这N个元素处理个遍
        i = 0
        while i < N:  # i是N个元素中的第i个
            for j in range(0, n): # 每轮的步长是n，j表示n步长内的操作
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]  # ^按位异或
            i = i+2*n
        n = 2*n  # 步长n逐层翻倍
    return x


A = polar_design_awgn(code_n, code_k, snr_dB)
A_matrix = A.reshape(-1, NN_size)
print('8个分为一组冻结位有多少个', '\n', A_matrix)
A_zeros = np.zeros([A_matrix.shape[0], A_matrix.shape[1]], dtype=np.float32)
A_count = (A_matrix - A_zeros).sum(axis=1)
# A_count = A_count.tolist()
print('每一行的信息位有多少个', '\n', A_count)

# 输出所有1,4,7,8 的位置，用于分类训练数据


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

    net_dict = {}
    batch_size = numOfWordSim * len(validation_snr)
    # 直接把X放进去译码,译码到倒数第四个数据层
    for i in range(0, N_LAYERS, 1):
        net_dict["hidden_right_x_0{0}".format(i)] = tf.slice(X, [0, 0], [batch_size, code_n])
        net_dict["hidden_left_x_0{0}".format(i)] = np.zeros([batch_size, code_n], dtype=np.float32)
        net_dict["hidden_right_x_1{0}".format(i)] = net_dict["hidden_left_x_{0}".format(int((n - 1) * (int(i / (n - 1))) - i % (n - 1) - 1))]

        net_dict["hidden_right_x_2{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_0{0}".format(i)],
                                                               [batch_size * 2 ** (int(i % (n - 1))), -1])
        net_dict["hidden_right_x_3{0}".format(i)] = tf.slice(net_dict["hidden_right_x_2{0}".format(i)], [0, 0],
                                                             [batch_size * 2 ** (int(i % (n - 1))),
                                                              int(2 ** (n - i % (n - 1) - 1))])
        net_dict["hidden_right_x_4{0}".format(i)] = tf.tile(net_dict["hidden_right_x_3{0}".format(i)], multiples=[1, 2])
        net_dict["hidden_right_x_5{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_4{0}".format(i)],
                                                               [batch_size, int(code_n)])  # L_i,j

        net_dict["hidden_right_x_6{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_1{0}".format(i)],
                                                               [batch_size * 2 ** (int(i % (n - 1))), -1])
        net_dict["hidden_right_x_7{0}".format(i)] = tf.slice(net_dict["hidden_right_x_6{0}".format(i)],
                                                             [0, 2 ** int((n - i % (n - 1) - 1))],
                                                             [batch_size * 2 ** (int(i % (n - 1))),
                                                              2 ** int((n - i % (n - 1) - 1))])
        net_dict["hidden_right_x_8{0}".format(i)] = tf.zeros(
            shape=[batch_size * 2 ** (int(i % (n - 1))), 2 ** int((n - i % (n - 1) - 1))], dtype=tf.float32)
        net_dict["hidden_right_x_9{0}".format(i)] = tf.slice(net_dict["hidden_right_x_6{0}".format(i)], [0, 0],
                                                             [batch_size * 2 ** (int(i % (n - 1))),
                                                              2 ** int((n - i % (n - 1) - 1))])
        net_dict["hidden_right_x_10{0}".format(i)] = tf.concat(
            [net_dict["hidden_right_x_7{0}".format(i)], net_dict["hidden_right_x_9{0}".format(i)]], 1)
        net_dict["hidden_right_x_11{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_10{0}".format(i)],
                                                                [batch_size, int(code_n)])  # [R_i+n/2^j, R_i]

        net_dict["hidden_right_x_12{0}".format(i)] = tf.slice(net_dict["hidden_right_x_2{0}".format(i)],
                                                              [0, 2 ** int((n - i % (n - 1) - 1))],
                                                              [batch_size * 2 ** (int(i % (n - 1))),
                                                               2 ** int((n - i % (n - 1) - 1))])
        net_dict["hidden_right_x_13{0}".format(i)] = tf.concat(
            [net_dict["hidden_right_x_8{0}".format(i)], net_dict["hidden_right_x_12{0}".format(i)]], 1)  # [0,R_i+n/2^j]
        net_dict["hidden_right_x_14{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_13{0}".format(i)],
                                                                [batch_size, int(code_n)])  # [0,L_i+n/2^j]

        net_dict["hidden_right_x_15{0}".format(i)] = tf.concat(
            [net_dict["hidden_right_x_12{0}".format(i)], net_dict["hidden_right_x_8{0}".format(i)]], 1)  # [R_i+n/2^j,0]
        net_dict["hidden_right_x_15{0}".format(i)] = tf.reshape(net_dict["hidden_right_x_15{0}".format(i)],
                                                                [batch_size, int(code_n)])  # [L_i+n/2^j,0]

        net_dict["hidden_right_x_16{0}".format(i)] = tf.sign(net_dict["hidden_right_x_5{0}".format(i)])  # sign(L_i,j)
        net_dict["hidden_right_x_17{0}".format(i)] = tf.add(net_dict["hidden_right_x_11{0}".format(i)],
                                                            net_dict["hidden_right_x_15{0}".format(i)])
        net_dict["hidden_right_x_18{0}".format(i)] = tf.sign(net_dict["hidden_right_x_17{0}".format(i)])  # sign
        net_dict["hidden_right_x_19{0}".format(i)] = tf.multiply(net_dict["hidden_right_x_16{0}".format(i)],
                                                                 net_dict["hidden_right_x_18{0}".format(i)])
        net_dict["hidden_right_x_20{0}".format(i)] = tf.where(
            tf.greater(tf.abs(net_dict["hidden_right_x_5{0}".format(i)]),
                       tf.abs(net_dict["hidden_right_x_17{0}".format(i)])),
            tf.abs(net_dict["hidden_right_x_17{0}".format(i)]),
            tf.abs(net_dict["hidden_right_x_5{0}".format(i)]))
        net_dict["hidden_right_x_21{0}".format(i)] = tf.multiply(net_dict["hidden_right_x_19{0}".format(i)],
                                                                 net_dict["hidden_right_x_20{0}".format(i)])
        net_dict["hidden_right_x_{0}".format(i)] = tf.add(net_dict["hidden_right_x_22{0}".format(i)],
                                                          net_dict["hidden_right_x_14{0}".format(i)])
    nn_input = net_dict["hidden_right_x_{0}".format(i)]

    return nn_input


check = create_mix_epoch_validation(code_k, code_n, numOfWordSim, VALIDATION_SNR, train_on_zero_word)
print(check.size)
