# -*- coding: utf-8 -*-



from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import random
import os
import regex
from collections import Counter



def split_data():
    '''
    划分数据

    '''

    cn_file = "./corpora/cn.txt"
    en_file = "./corpora/en.txt"

    train_cn_file = "./corpora/train_cn"
    train_en_file = "./corpora/train_en"
    test_cn_file = "./corpora/test_cn"
    test_en_file = "./corpora/test_en"

    #读取中英文数据
    with codecs.open(cn_file, "r", "utf-8") as cn, codecs.open(en_file, "r", "utf-8") as en:
        cn_texts = cn.readlines()
        en_texts = en.readlines()

    length = len(cn_texts)
    ratio = 0.7 #训练数据比例
    num = int(length * ratio)

    train_idx = random.sample(range(length), num)
    max_cn_len = 0 #中文最大长度
    max_en_len = 0 #英文最大长度

    #划分训练集测试集
    with codecs.open(train_cn_file, "w", "utf-8") as train_cn, codecs.open(train_en_file, "w", "utf-8") as train_en, codecs.open(test_cn_file, "w", "utf-8") as test_cn, codecs.open(test_en_file, "w", "utf-8") as test_en:
        for i in range(length):
            if i in train_idx:
                train_cn.write(cn_texts[i])
                train_en.write(en_texts[i])
            else:
                test_cn.write(cn_texts[i])
                test_en.write(en_texts[i])
        
            if len(cn_texts[i]) > max_cn_len:
                max_cn_len = len(cn_texts[i])
            
            if len(en_texts[i]) > max_en_len:
                max_en_len = len(en_texts[i])
            
    print(max_cn_len, max_en_len)


def make_src_vocab(fpath, fname):
    '''构建中文词库.
    
    Args:
      fpath: 输入文件名.
      fname: 输出文件名.
    
    '''  
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Han}']", "", text) #提取中文词
    words = text.split()
    word2cnt = Counter(words)   #统计词量
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))
            
def make_tgt_vocab(fpath, fname):
    '''构建英文词库.
    
    Args:
      fpath: 输入文件名.
      fname: 输出文件名.
    
    '''  
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)   #提取英文单词
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    split_data()
    make_src_vocab(hp.source_train, "cn.vocab.tsv")
    make_tgt_vocab(hp.target_train, "en.vocab.tsv")
