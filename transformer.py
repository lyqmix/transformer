# -*- coding: utf-8 -*-



from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from utils import *
import os, codecs
from tqdm import tqdm

import numpy as np

from hyperparams import Hyperparams as hp

from nltk.translate.bleu_score import corpus_bleu

import time

class Transformer():
    def __graph(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data() 
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # 解码端输入
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

            # 加载词汇表  
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
            
            # 编码端
            with tf.variable_scope("encoder"):
                ## 词向量
                self.enc = embedding(self.x, 
                                      vocab_size=len(de2idx), 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_embed")
                
                ## 位置向量
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ##dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## 编码端含有的编码块数，每个编码块包含一个多头自注意力模块和前向模块
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### 多头自注意力
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### 前向
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            
            # 解码端
            with tf.variable_scope("decoder"):
                ## 词向量
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(en2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## 位置向量
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ##解码端含有的解码块数，每个解码块包含一个多头自注意力模块，多头编码解码注意力模块和前向模块 
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## 多头自注意力
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## 多头编码解码注意力
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## 前向
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            #线性投射
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
            if is_training:  
                # 损失函数
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # 训练
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

    def train(self):
            # 加载词汇表 
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab() 
        # 构建计算图
        self.__graph("train")
        print('计算图加载完毕')
        # 开启会话
        sv = tf.train.Supervisor(graph=self.graph, logdir=hp.logdir,save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(self.num_batch), total=self.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(self.train_op)
                gs = sess.run(self.global_step)   
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                print('/model_epoch_%02d_gs_%d' % (epoch, gs))

        print('结束')         
       

    def eval(self): 
        # 加载计算图
        self.__graph(is_training=False)
        print("计算图加载完毕")
    
        # 加载数据
        X, Sources, Targets = load_test_data()
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()

        # 开启会话         
        with self.graph.as_default():    
            sv = tf.train.Supervisor()
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ## 恢复参数
                sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
                print("恢复!")
              
                ## 获取模型名字
                mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # 模型名称
             
                ## 前向推导
                if not os.path.exists('results'): os.mkdir('results')
                with codecs.open("results/" + mname, "w", "utf-8") as fout:
                    list_of_refs, hypotheses = [], []
                    epoch = len(X) // hp.batch_size
                    for i in range(epoch):
                        start = time.time()
                     
                        ### 获取 mini-batches
                        x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                        sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                        targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                        ### 自回归推理
                        preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                        for j in range(hp.maxlen):
                            _preds = sess.run(self.preds, {self.x: x, self.y: preds})
                            preds[:, j] = _preds[:, j]
                     
                        ### 保存
                        for source, target, pred in zip(sources, targets, preds):
                            got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                
                            fout.write("- source: " + source +"\n")
                            fout.write("- expected: " + target + "\n")
                            fout.write("- got: " + got + "\n\n")
                            fout.flush()
                        
                            print(source)
                            print(target)
                            print(got)
                          
                            # bleu 值
                            ref = target.split()
                            hypothesis = got.split()
                            if len(ref) > 3 and len(hypothesis) > 3:
                                list_of_refs.append([ref])
                                hypotheses.append(hypothesis)
                        batch_time = time.time() - start
                        print("i = {} / {}, time = {}s, remain = {}s".format(i, epoch, batch_time, (epoch-i)*batch_time))

                    ## 计算bleu值
                    score = corpus_bleu(list_of_refs, hypotheses)
                    fout.write("Bleu Score = " + str(100*score))
        print('结束')


          
