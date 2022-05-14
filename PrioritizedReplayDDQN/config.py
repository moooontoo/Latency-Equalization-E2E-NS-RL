# capacity = 8000
# learning_rate = 1e-3
# memory_count = 0
# batch_size = 256
# gamma = 0.995
# update_count = 0
# coding=utf-8
import numpy as np

#replay_memory所需的相关参数
class AgentConfig(object):
    kesi = 0.1  # kesi是随机选动作
    heigth = 1
    length = 1
    width = 20
    # fc_out_dims = 1000

    # learning_rate = 0.0025
    # learning_rate = 3e-4
    LEARNING_RATE = 1e-3
    learning_rate_policy = 5e-3
    learning_rate_value = 1e-3
    learning_rate_q = 1e-3
    # learning_rate_minimum = 0.0025
    # learning_rate_decay = 0.96
    # learning_rate_decay_step = 5000
    # discount = 0.9
    max_step = 30000  # 训练次数
    learn_start = 1000  # 1000次之后网络开始更新
    train_frequency = 4  # 1000次之后每4个动作更新一下训练网络
    # train_step = 2000
    train_step = 2000 # 暂时每100次print一次
    test_step = 2000  # 2000次判断一下是否保存模型，然后对数据进行summary
    target_q_update_step = 1000  # 1000次之后，每1000次更新一下目标网络
    play_step = 500  # 每3000次看下模型调节能力
    play_times = 30

    batch_size = 32

    memory_size = 2000
    test_play = 500
    # test_produce =

    action_rows = 20 #20个次用户
    action_cols = 20 #20个信道
    # action_size = 44
    reward_size = 1




#神经网络所需参数
class NetworkConfig(object):
    mbs_number = 1 # macro基站数量：1
    sbs_number = 1 # small基站数量：1
    u_number = 20 # 用户数量：20
    c_num = 20 # 信道数量
    BW_sc = 18e4 # 每个信道的带宽
    sp_max = 1 # 基站发射功率最大值
    N0 = 5e-22 # AWGN噪声
    net_size = (100, 100) # 网络尺寸
    #E2E delay limit
    latency_limit = 10
    latency_limit2 = 240
    rate_limit = 2e5
    reward_alpha = 0