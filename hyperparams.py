# -*- coding: utf-8 -*-



class Hyperparams:
    '''项目所有的超参数'''
    # 数据超参
    source_train = 'corpora/train_cn'
    target_train = 'corpora/train_en'
    source_test = 'corpora/test_cn'
    target_test = 'corpora/test_en'
    
    # 训练超参
    batch_size = 32 # 批量随机梯度
    lr = 0.0001 # 学习率
    logdir = 'logdir' # 模型保存目录
    
    # 模型超参
    maxlen = 100    #句子中词的最大数量
    min_cnt = 20    #少于min_cnt数量的单词一律统计为<UNK>.
    hidden_units = 512  
    num_blocks = 6  #编码解码块大小
    num_epochs = 60 #训练轮数
    num_heads = 8   #多头数
    dropout_rate = 0.1 
    sinusoid = False    #使用位置词向量.
    
    
    
    
