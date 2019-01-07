# -*- coding: utf-8 -*-



from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import regex


def load_de_vocab():
    '''
    构建双向中文词典

    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/cn.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    '''
    构建双向英文词典

    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    '''
    创建数据

    '''
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # 忽略句子长度大于hp.maxlen的数据
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] 
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # 填充
    X = np.zeros([len(x_list), hp.maxlen], np.int32) #中文id数组 
    Y = np.zeros([len(y_list), hp.maxlen], np.int32) #英文id数组
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    '''
    加载训练数据

    '''
    de_sents = [regex.sub("[^\s\p{Han}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y
    
def load_test_data():
    '''
    加载测试数据

    '''
    def _refine1(line):
        line = regex.sub("[^\s\p{Han}']", "", line) 
        return line.strip()

    def _refine2(line):
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine1(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    en_sents = [_refine2(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
        

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets 

def get_batch_data():
    # 加载数据
    X, Y = load_train_data()
    
    # 一个epoch含有num_batch个batch
    num_batch = len(X) // hp.batch_size
    
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    #tensor生成器
    input_queues = tf.train.slice_input_producer([X, Y])
            
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch 


def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''
    数据标准化

    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''
    构建词向量
       
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''
    构建位置向量

    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])


        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  

        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs



def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''
      多头注意力计算

    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # 生成Q K V 矩阵
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # 将 Q K V 矩阵切分拼接（多头）
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # 点积
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # 缩放
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        #密钥掩蔽
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # 归一化
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
       
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # 计算注意力结果
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # 残差
        outputs += queries
              
        # 标准化
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''
    前馈网络计算

    '''
    with tf.variable_scope(scope, reuse=reuse):
       
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
       
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        outputs += inputs

        outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
 
    K = inputs.get_shape().as_list()[-1] 
    return ((1-epsilon) * inputs) + (epsilon / K)
    
    

            
