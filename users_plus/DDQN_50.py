import pickle
from collections import namedtuple

import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from users_plus.network_50 import Network
from users_plus.DeepQNetwork_50 import *
from users_plus.config_50 import *
from users_plus.replay_memory import ReplayMemory
from users_plus.utils import *
from numpy import random
import copy
num = 50 #总人数
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
#随机选信道、信道上功率
def random_select_action():
    #np.random.ranint(a,b)-->[a,b)
    #生成[0,20)上取值为随机整数，1x20的向量
    x_c = np.random.randint(0, num, size=num)
    x_p = np.random.randint(0, num, size=num)
    action_c = np.zeros((num, num))
    action_pc = np.zeros((num, num))

    for i in range(num):
        action_c[i][x_c[i]] = 1
        action_pc[i][x_c[i]] = x_p[i] * 2
        #每个等级2W
    return action_pc, action_c


#act_net()用于生成动作
def select_action(state,act_net):
    # 参数 state:1*3
    state = torch.from_numpy(state).float()
    x1_p, x1_c, x2_p, x2_c, x3_p, x3_c, x4_p, x4_c, x5_p, x5_c, x6_p, x6_c, x7_p, x7_c, x8_p, x8_c, x9_p, x9_c, x10_p, x10_c, x11_p, \
    x11_c, x12_p, x12_c, x13_p, x13_c, x14_p, x14_c, x15_p, x15_c, x16_p, x16_c, x17_p, x17_c, x18_p, x18_c, x19_p, x19_c, x20_p, x20_c, \
    x21_p, x21_c, x22_p, x22_c, x23_p, x23_c, x24_p, x24_c, x25_p, x25_c, x26_p, x26_c, x27_p, x27_c, x28_p, x28_c, x29_p, x29_c, x30_p, x30_c,\
    x31_p, x31_c, x32_p, x32_c, x33_p, x33_c, x34_p, x34_c, x35_p, x35_c, x36_p, x36_c, x37_p, x37_c, x38_p, x38_c, x39_p, x39_c, x40_p, x40_c,\
    x41_p, x41_c, x42_p, x42_c, x43_p, x43_c, x44_p, x44_c, x45_p, x45_c, x46_p, x46_c, x47_p, x47_c, x48_p, x48_c, x49_p, x49_c, x50_p, x50_c = act_net(state)
    # 功率选择为20个功率等级
    x1_p = np.argmax(x1_p.view(num).cpu().detach().numpy())
    x2_p = np.argmax(x2_p.view(num).cpu().detach().numpy())
    x3_p = np.argmax(x3_p.view(num).cpu().detach().numpy())
    x4_p = np.argmax(x4_p.view(num).cpu().detach().numpy())
    x5_p = np.argmax(x5_p.view(num).cpu().detach().numpy())
    x6_p = np.argmax(x6_p.view(num).cpu().detach().numpy())
    x7_p = np.argmax(x7_p.view(num).cpu().detach().numpy())
    x8_p = np.argmax(x8_p.view(num).cpu().detach().numpy())
    x9_p = np.argmax(x9_p.view(num).cpu().detach().numpy())
    x10_p = np.argmax(x10_p.view(num).cpu().detach().numpy())
    x11_p = np.argmax(x11_p.view(num).cpu().detach().numpy())
    x12_p = np.argmax(x12_p.view(num).cpu().detach().numpy())
    x13_p = np.argmax(x13_p.view(num).cpu().detach().numpy())
    x14_p = np.argmax(x14_p.view(num).cpu().detach().numpy())
    x15_p = np.argmax(x15_p.view(num).cpu().detach().numpy())
    x16_p = np.argmax(x16_p.view(num).cpu().detach().numpy())
    x17_p = np.argmax(x17_p.view(num).cpu().detach().numpy())
    x18_p = np.argmax(x18_p.view(num).cpu().detach().numpy())
    x19_p = np.argmax(x19_p.view(num).cpu().detach().numpy())
    x20_p = np.argmax(x20_p.view(num).cpu().detach().numpy())
    x21_p = np.argmax(x21_p.view(num).cpu().detach().numpy())
    x22_p = np.argmax(x22_p.view(num).cpu().detach().numpy())
    x23_p = np.argmax(x23_p.view(num).cpu().detach().numpy())
    x24_p = np.argmax(x24_p.view(num).cpu().detach().numpy())
    x25_p = np.argmax(x25_p.view(num).cpu().detach().numpy())
    x26_p = np.argmax(x26_p.view(num).cpu().detach().numpy())
    x27_p = np.argmax(x27_p.view(num).cpu().detach().numpy())
    x28_p = np.argmax(x28_p.view(num).cpu().detach().numpy())
    x29_p = np.argmax(x29_p.view(num).cpu().detach().numpy())
    x30_p = np.argmax(x30_p.view(num).cpu().detach().numpy())
    x31_p = np.argmax(x31_p.view(num).cpu().detach().numpy())
    x32_p = np.argmax(x32_p.view(num).cpu().detach().numpy())
    x33_p = np.argmax(x33_p.view(num).cpu().detach().numpy())
    x34_p = np.argmax(x34_p.view(num).cpu().detach().numpy())
    x35_p = np.argmax(x35_p.view(num).cpu().detach().numpy())
    x36_p = np.argmax(x36_p.view(num).cpu().detach().numpy())
    x37_p = np.argmax(x37_p.view(num).cpu().detach().numpy())
    x38_p = np.argmax(x38_p.view(num).cpu().detach().numpy())
    x39_p = np.argmax(x39_p.view(num).cpu().detach().numpy())
    x40_p = np.argmax(x40_p.view(num).cpu().detach().numpy())
    x41_p = np.argmax(x41_p.view(num).cpu().detach().numpy())
    x42_p = np.argmax(x42_p.view(num).cpu().detach().numpy())
    x43_p = np.argmax(x43_p.view(num).cpu().detach().numpy())
    x44_p = np.argmax(x44_p.view(num).cpu().detach().numpy())
    x45_p = np.argmax(x45_p.view(num).cpu().detach().numpy())
    x46_p = np.argmax(x46_p.view(num).cpu().detach().numpy())
    x47_p = np.argmax(x47_p.view(num).cpu().detach().numpy())
    x48_p = np.argmax(x48_p.view(num).cpu().detach().numpy())
    x49_p = np.argmax(x49_p.view(num).cpu().detach().numpy())
    x50_p = np.argmax(x50_p.view(num).cpu().detach().numpy())
    x_p = np.array(
        [x1_p, x2_p, x3_p, x4_p, x5_p, x6_p, x7_p, x8_p, x9_p, x10_p, x11_p, x12_p, x13_p, x14_p, x15_p, x16_p, x17_p,
         x18_p, x19_p, x20_p,x21_p, x22_p, x23_p, x24_p, x25_p, x26_p, x27_p, x28_p, x29_p, x30_p,
         x31_p, x32_p, x33_p, x34_p, x35_p, x36_p, x37_p, x38_p, x39_p, x40_p,x41_p, x42_p, x43_p, x44_p, x45_p, x46_p, x47_p, x48_p, x49_p, x50_p])
    # 信道选择为6个可选信道
    x1_c = np.argmax(x1_c.view(num).cpu().detach().numpy())
    x2_c = np.argmax(x2_c.view(num).cpu().detach().numpy())
    x3_c = np.argmax(x3_c.view(num).cpu().detach().numpy())
    x4_c = np.argmax(x4_c.view(num).cpu().detach().numpy())
    x5_c = np.argmax(x5_c.view(num).cpu().detach().numpy())
    x6_c = np.argmax(x6_c.view(num).cpu().detach().numpy())
    x7_c = np.argmax(x7_c.view(num).cpu().detach().numpy())
    x8_c = np.argmax(x8_c.view(num).cpu().detach().numpy())
    x9_c = np.argmax(x9_c.view(num).cpu().detach().numpy())
    x10_c = np.argmax(x10_c.view(num).cpu().detach().numpy())
    x11_c = np.argmax(x11_c.view(num).cpu().detach().numpy())
    x12_c = np.argmax(x12_c.view(num).cpu().detach().numpy())
    x13_c = np.argmax(x13_c.view(num).cpu().detach().numpy())
    x14_c = np.argmax(x14_c.view(num).cpu().detach().numpy())
    x15_c = np.argmax(x15_c.view(num).cpu().detach().numpy())
    x16_c = np.argmax(x16_c.view(num).cpu().detach().numpy())
    x17_c = np.argmax(x17_c.view(num).cpu().detach().numpy())
    x18_c = np.argmax(x18_c.view(num).cpu().detach().numpy())
    x19_c = np.argmax(x19_c.view(num).cpu().detach().numpy())
    x20_c = np.argmax(x20_c.view(num).cpu().detach().numpy())
    x21_c = np.argmax(x21_c.view(num).cpu().detach().numpy())
    x22_c = np.argmax(x22_c.view(num).cpu().detach().numpy())
    x23_c = np.argmax(x23_c.view(num).cpu().detach().numpy())
    x24_c = np.argmax(x24_c.view(num).cpu().detach().numpy())
    x25_c = np.argmax(x25_c.view(num).cpu().detach().numpy())
    x26_c = np.argmax(x26_c.view(num).cpu().detach().numpy())
    x27_c = np.argmax(x27_c.view(num).cpu().detach().numpy())
    x28_c = np.argmax(x28_c.view(num).cpu().detach().numpy())
    x29_c = np.argmax(x29_c.view(num).cpu().detach().numpy())
    x30_c = np.argmax(x30_c.view(num).cpu().detach().numpy())
    x31_c = np.argmax(x31_c.view(num).cpu().detach().numpy())
    x32_c = np.argmax(x32_c.view(num).cpu().detach().numpy())
    x33_c = np.argmax(x33_c.view(num).cpu().detach().numpy())
    x34_c = np.argmax(x34_c.view(num).cpu().detach().numpy())
    x35_c = np.argmax(x35_c.view(num).cpu().detach().numpy())
    x36_c = np.argmax(x36_c.view(num).cpu().detach().numpy())
    x37_c = np.argmax(x37_c.view(num).cpu().detach().numpy())
    x38_c = np.argmax(x38_c.view(num).cpu().detach().numpy())
    x39_c = np.argmax(x39_c.view(num).cpu().detach().numpy())
    x40_c = np.argmax(x40_c.view(num).cpu().detach().numpy())
    x41_c = np.argmax(x41_c.view(num).cpu().detach().numpy())
    x42_c = np.argmax(x42_c.view(num).cpu().detach().numpy())
    x43_c = np.argmax(x43_c.view(num).cpu().detach().numpy())
    x44_c = np.argmax(x44_c.view(num).cpu().detach().numpy())
    x45_c = np.argmax(x45_c.view(num).cpu().detach().numpy())
    x46_c = np.argmax(x46_c.view(num).cpu().detach().numpy())
    x47_c = np.argmax(x47_c.view(num).cpu().detach().numpy())
    x48_c = np.argmax(x48_c.view(num).cpu().detach().numpy())
    x49_c = np.argmax(x49_c.view(num).cpu().detach().numpy())
    x50_c = np.argmax(x50_c.view(num).cpu().detach().numpy())
    x_c = np.array(
        [x1_c, x2_c, x3_c, x4_c, x5_c, x6_c, x7_c, x8_c, x9_c, x10_c, x11_c, x12_c, x13_c, x14_c, x15_c, x16_c, x17_c,
         x18_c, x19_c, x20_c,x21_c, x22_c, x23_c, x24_c, x25_c, x26_c, x27_c, x28_c, x29_c, x30_c,
         x31_c, x32_c, x33_c, x34_c, x35_c, x36_c, x37_c, x38_c, x39_c, x40_c,x41_c, x42_c, x43_c, x44_c, x45_c, x46_c, x47_c, x48_c, x49_c, x50_c])

    action_c = np.zeros((num, num))
    action_pc = np.zeros((num, num))

    for i in range(num):
        action_c[i][x_c[i]] = 1
        action_pc[i][x_c[i]] = x_p[i] * 2   #每个功率等级2W

    if np.random.rand(1) >= 0.9:  # epslion greedy
        action_pc, action_c = random_select_action()  # epslion greedy

    return action_pc, action_c  # 返回用户信道 功率分配情况


def rlRan(sla,target_net,act_net,max_episodes = 5000):
    #Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])



    learning_rate = 0.001  # 经过实验发现learning_rate取0.001收敛速度很快，0.01很慢，0.005也还可以
    gamma = 0.995
    #episodes = 1000
    wave = []

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # target_net, act_net = Net(), Net()
    # if os.path.exists('../models/act_ddqn'):
    #     act_net.load_state_dict(torch.load('../models/act_ddqn'))
    #     target_net.load_state_dict(torch.load('../models/target_ddqn'))
    optimizer = optim.Adam(act_net.parameters(), learning_rate)

    #Tools.create_dirs()
    cong = NetworkConfig
    #物理网络所需的若干功能，比如计算信道增益、速率等等
    net = Network(cong)
    memory = ReplayMemory(AgentConfig)
    # max_avg_ep_reward = 0.
    ep_rewards = []
    ep_se = []
    ep_qoe = []
    ep_latency = np.array([0] * num)
    ep_ssl = []
    ep_es = []
    ep_us = []

    s = net.state_first()  # 先代入初始状态   之后进行更新
    s = np.array(s).reshape(1, num)

    se = 0.
    qoe = 0.
    for i_episode in range(max_episodes):
        # choose action
        a_p, a_c = select_action(s,act_net)  # 动作：20个次用户的信道分配以及功率分配

        ##将20个用户的数据率作为状态
        s_ = net.u_rate(a_p, a_c)
        s_ = np.array(s_).reshape(1, num)

        # compute reward
        r, se, qoe, latency, ssl, es, us = net.compute_reward(a_p, a_c, sla)
        ep_rewards.append(r)
        ep_se.append(se)
        ep_qoe.append(qoe)
        ep_latency = ep_latency + latency
        ep_ssl.append(ssl)
        ep_es.append(es)
        ep_us.append(us)
        memory.add(s, r, a_p)
        s = s_  # 状态转移

        # #前200个回合 进行试探
        # if (i_episode + 1) <= 200 and (i_episode + 1) % 100 == 0:
        #     ep_rewards = []
        #     se = []
        #     qoe = []
        # # kkw
        # # end 存储
        if (i_episode + 1) <= 200 and (i_episode + 1) % 100 == 0:
            avg_ep_reward = np.mean(ep_rewards)
            avg_ep_se = np.mean(ep_se)
            avg_ep_qoe = np.mean(ep_qoe)
            avg_ep_latency = ep_latency / 100
            avg_ep_ssl = np.mean(ep_ssl)
            avg_es = np.mean(ep_es)
            avg_us = np.mean(ep_us)
            print('step:', (i_episode + 1), 'avg_reward:', avg_ep_reward, 'avg_se:', avg_ep_se, 'avg_qoe:', avg_ep_qoe,
                  'avg_ssl:', avg_ep_ssl, 'embb', avg_es, 'urllc', avg_us)
            ep_rewards = []
            ep_se = []
            ep_qoe = []
            ep_latency = np.array([0] * num)
            ep_ssl = []
            ep_es = []
            ep_us = []



        if (i_episode + 1) > 200:  # 原2000，原200，还是200比较合适
            # kkw
            bn_s, bn_a, bn_r, bn_s_ = memory.sample()  # 先放一放这个
            bn_s = torch.tensor(bn_s).float().to(device)  # [batch_size, 1, 50, 1]
            bn_a = torch.tensor(bn_a).float().to(device)  # [batch_size, 10, 10]
            bn_r = torch.tensor(bn_r).float().to(device)  # [batch_size, 1]
            bn_s_ = torch.tensor(bn_s_).float().to(device)  # [batch_size, 1, 50, 1]

            # reward = (bn_r - bn_r.mean()) / (reward.std() + 1e-7)

            x1_p, x1_c, x2_p, x2_c, x3_p, x3_c, x4_p, x4_c, x5_p, x5_c, x6_p, x6_c, x7_p, x7_c, x8_p, x8_c, x9_p, x9_c, x10_p, x10_c,x11_p, x11_c, x12_p, x12_c, x13_p, x13_c, x14_p, x14_c, x15_p, x15_c, x16_p, x16_c, x17_p, x17_c, x18_p, x18_c, x19_p, x19_c, x20_p, x20_c,x21_p, x21_c, x22_p, x22_c, x23_p, x23_c, x24_p, x24_c, x25_p, x25_c, x26_p, x26_c, x27_p, x27_c, x28_p, x28_c, x29_p, x29_c, x30_p, x30_c, x31_p, x31_c, x32_p, x32_c, x33_p, x33_c, x34_p, x34_c, x35_p, x35_c, x36_p, x36_c, x37_p, x37_c, x38_p, x38_c, x39_p, x39_c, x40_p, x40_c,x41_p, x41_c, x42_p, x42_c, x43_p, x43_c, x44_p, x44_c, x45_p, x45_c, x46_p, x46_c, x47_p, x47_c, x48_p, x48_c, x49_p, x49_c, x50_p, x50_c= act_net( bn_s )
            x1_p_t, x1_c_t, x2_p_t, x2_c_t, x3_p_t, x3_c_t, x4_p_t, x4_c_t, x5_p_t, x5_c_t, x6_p_t, x6_c_t, x7_p_t, x7_c_t, x8_p_t, x8_c_t, x9_p_t, x9_c_t, x10_p_t, x10_c_t,\
            x11_p_t, x11_c_t, x12_p_t, x12_c_t, x13_p_t, x13_c_t, x14_p_t, x14_c_t, x15_p_t, x15_c_t, x16_p_t, x16_c_t, x17_p_t, x17_c_t, x18_p_t, x18_c_t,\
            x19_p_t, x19_c_t, x20_p_t, x20_c_t, x21_p_t, x21_c_t, x22_p_t, x22_c_t, x23_p_t, x23_c_t, x24_p_t, x24_c_t, x25_p_t, x25_c_t,\
            x26_p_t, x26_c_t, x27_p_t, x27_c_t, x28_p_t, x28_c_t, x29_p_t, x29_c_t, x30_p_t, x30_c_t, \
            x31_p_t, x31_c_t, x32_p_t, x32_c_t, x33_p_t, x33_c_t, x34_p_t, x34_c_t, x35_p_t, x35_c_t, x36_p_t, x36_c_t, x37_p_t, x37_c_t, x38_p_t, x38_c_t, x39_p_t, x39_c_t, x40_p_t, x40_c_t, \
            x41_p_t, x41_c_t, x42_p_t, x42_c_t, x43_p_t, x43_c_t, x44_p_t, x44_c_t, x45_p_t, x45_c_t, x46_p_t, x46_c_t, x47_p_t, x47_c_t, x48_p_t, x48_c_t, x49_p_t, x49_c_t, x50_p_t, x50_c_t = target_net( bn_s_ )

            # x = torch.zeros(32, 1, 40)
            # x_t = torch.zeros(32, 1, 40)
            x = torch.zeros(32, 1, 2*num)
            x_t = torch.zeros(32, 1, 2*num)

            for i in range(32):
                action_no = np.argmax(bn_a[i][0])
                action_p = bn_a[i][0][action_no]
                x[i][0][0] = x1_c[i][0][action_no]
                x[i][0][1] = x1_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][1])
                action_p = bn_a[i][0][action_no]
                x[i][0][2] = x2_c[i][0][action_no]
                x[i][0][3] = x2_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][2])
                action_p = bn_a[i][0][action_no]
                x[i][0][4] = x3_c[i][0][action_no]
                x[i][0][5] = x3_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][3])
                action_p = bn_a[i][0][action_no]
                x[i][0][6] = x4_c[i][0][action_no]
                x[i][0][7] = x4_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][4])
                action_p = bn_a[i][0][action_no]
                x[i][0][8] = x5_c[i][0][action_no]
                x[i][0][9] = x5_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][5])
                action_p = bn_a[i][0][action_no]
                x[i][0][10] = x6_c[i][0][action_no]
                x[i][0][11] = x6_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][6])
                action_p = bn_a[i][0][action_no]
                x[i][0][12] = x7_c[i][0][action_no]
                x[i][0][13] = x7_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][7])
                action_p = bn_a[i][0][action_no]
                x[i][0][14] = x8_c[i][0][action_no]
                x[i][0][15] = x8_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][8])
                action_p = bn_a[i][0][action_no]
                x[i][0][16] = x9_c[i][0][action_no]
                x[i][0][17] = x9_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][9])
                action_p = bn_a[i][0][action_no]
                x[i][0][18] = x10_c[i][0][action_no]
                x[i][0][19] = x10_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][10])
                action_p = bn_a[i][0][action_no]
                x[i][0][20] = x11_c[i][0][action_no]
                x[i][0][21] = x11_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][11])
                action_p = bn_a[i][0][action_no]
                x[i][0][22] = x12_c[i][0][action_no]
                x[i][0][23] = x12_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][12])
                action_p = bn_a[i][0][action_no]
                x[i][0][24] = x13_c[i][0][action_no]
                x[i][0][25] = x13_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][13])
                action_p = bn_a[i][0][action_no]
                x[i][0][26] = x14_c[i][0][action_no]
                x[i][0][27] = x14_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][14])
                action_p = bn_a[i][0][action_no]
                x[i][0][28] = x15_c[i][0][action_no]
                x[i][0][29] = x15_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][15])
                action_p = bn_a[i][0][action_no]
                x[i][0][30] = x16_c[i][0][action_no]
                x[i][0][31] = x16_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][16])
                action_p = bn_a[i][0][action_no]
                x[i][0][32] = x17_c[i][0][action_no]
                x[i][0][33] = x17_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][17])
                action_p = bn_a[i][0][action_no]
                x[i][0][34] = x18_c[i][0][action_no]
                x[i][0][35] = x18_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][18])
                action_p = bn_a[i][0][action_no]
                x[i][0][36] = x19_c[i][0][action_no]
                x[i][0][37] = x19_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][19])
                action_p = bn_a[i][0][action_no]
                x[i][0][38] = x20_c[i][0][action_no]
                x[i][0][39] = x20_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][20])
                action_p = bn_a[i][0][action_no]
                x[i][0][40] = x21_c[i][0][action_no]
                x[i][0][41] = x21_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][21])
                action_p = bn_a[i][0][action_no]
                x[i][0][42] = x22_c[i][0][action_no]
                x[i][0][43] = x22_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][22])
                action_p = bn_a[i][0][action_no]
                x[i][0][44] = x23_c[i][0][action_no]
                x[i][0][45] = x23_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][23])
                action_p = bn_a[i][0][action_no]
                x[i][0][46] = x24_c[i][0][action_no]
                x[i][0][47] = x24_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][24])
                action_p = bn_a[i][0][action_no]
                x[i][0][48] = x25_c[i][0][action_no]
                x[i][0][49] = x25_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][25])
                action_p = bn_a[i][0][action_no]
                x[i][0][50] = x26_c[i][0][action_no]
                x[i][0][51] = x26_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][26])
                action_p = bn_a[i][0][action_no]
                x[i][0][52] = x27_c[i][0][action_no]
                x[i][0][53] = x27_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][27])
                action_p = bn_a[i][0][action_no]
                x[i][0][54] = x28_c[i][0][action_no]
                x[i][0][55] = x28_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][28])
                action_p = bn_a[i][0][action_no]
                x[i][0][56] = x29_c[i][0][action_no]
                x[i][0][57] = x29_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][29])
                action_p = bn_a[i][0][action_no]
                x[i][0][58] = x30_c[i][0][action_no]
                x[i][0][59] = x30_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][30])
                action_p = bn_a[i][0][action_no]
                x[i][0][60] = x31_c[i][0][action_no]
                x[i][0][61] = x31_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][31])
                action_p = bn_a[i][0][action_no]
                x[i][0][62] = x32_c[i][0][action_no]
                x[i][0][63] = x32_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][32])
                action_p = bn_a[i][0][action_no]
                x[i][0][64] = x33_c[i][0][action_no]
                x[i][0][65] = x33_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][33])
                action_p = bn_a[i][0][action_no]
                x[i][0][66] = x34_c[i][0][action_no]
                x[i][0][67] = x34_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][34])
                action_p = bn_a[i][0][action_no]
                x[i][0][68] = x35_c[i][0][action_no]
                x[i][0][69] = x35_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][35])
                action_p = bn_a[i][0][action_no]
                x[i][0][70] = x36_c[i][0][action_no]
                x[i][0][71] = x36_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][36])
                action_p = bn_a[i][0][action_no]
                x[i][0][72] = x37_c[i][0][action_no]
                x[i][0][73] = x37_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][37])
                action_p = bn_a[i][0][action_no]
                x[i][0][74] = x38_c[i][0][action_no]
                x[i][0][75] = x38_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][38])
                action_p = bn_a[i][0][action_no]
                x[i][0][76] = x39_c[i][0][action_no]
                x[i][0][77] = x39_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][39])
                action_p = bn_a[i][0][action_no]
                x[i][0][78] = x40_c[i][0][action_no]
                x[i][0][79] = x40_p[i][0][(action_p / 2).numpy()]

                action_p = bn_a[i][0][action_no]
                x[i][0][80] = x4_c[i][0][action_no]
                x[i][0][81] = x41_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][1])
                action_p = bn_a[i][0][action_no]
                x[i][0][82] = x42_c[i][0][action_no]
                x[i][0][83] = x42_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][2])
                action_p = bn_a[i][0][action_no]
                x[i][0][84] = x43_c[i][0][action_no]
                x[i][0][85] = x43_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][3])
                action_p = bn_a[i][0][action_no]
                x[i][0][86] = x44_c[i][0][action_no]
                x[i][0][87] = x44_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][4])
                action_p = bn_a[i][0][action_no]
                x[i][0][88] = x45_c[i][0][action_no]
                x[i][0][89] = x45_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][5])
                action_p = bn_a[i][0][action_no]
                x[i][0][90] = x46_c[i][0][action_no]
                x[i][0][91] = x46_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][6])
                action_p = bn_a[i][0][action_no]
                x[i][0][92] = x47_c[i][0][action_no]
                x[i][0][93] = x47_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][7])
                action_p = bn_a[i][0][action_no]
                x[i][0][94] = x48_c[i][0][action_no]
                x[i][0][95] = x48_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][8])
                action_p = bn_a[i][0][action_no]
                x[i][0][96] = x49_c[i][0][action_no]
                x[i][0][97] = x49_p[i][0][(action_p / 2).numpy()]

                action_no = np.argmax(bn_a[i][9])
                action_p = bn_a[i][0][action_no]
                x[i][0][98] = x50_c[i][0][action_no]
                x[i][0][99] = x50_p[i][0][(action_p / 2).numpy()]

                x_t[i][0][0] = x1_c_t[i][0].max()
                x_t[i][0][1] = x1_p_t[i][0].max()
                x_t[i][0][2] = x2_c_t[i][0].max()
                x_t[i][0][3] = x2_p_t[i][0].max()
                x_t[i][0][4] = x3_c_t[i][0].max()
                x_t[i][0][5] = x3_p_t[i][0].max()
                x_t[i][0][6] = x4_c_t[i][0].max()
                x_t[i][0][7] = x4_p_t[i][0].max()
                x_t[i][0][8] = x5_c_t[i][0].max()
                x_t[i][0][9] = x5_p_t[i][0].max()
                x_t[i][0][10] = x6_c_t[i][0].max()
                x_t[i][0][11] = x6_p_t[i][0].max()
                x_t[i][0][12] = x7_c_t[i][0].max()
                x_t[i][0][13] = x7_p_t[i][0].max()
                x_t[i][0][14] = x8_c_t[i][0].max()
                x_t[i][0][15] = x8_p_t[i][0].max()
                x_t[i][0][16] = x9_c_t[i][0].max()
                x_t[i][0][17] = x9_p_t[i][0].max()
                x_t[i][0][18] = x10_c_t[i][0].max()
                x_t[i][0][19] = x10_p_t[i][0].max()
                x_t[i][0][20] = x11_c_t[i][0].max()
                x_t[i][0][21] = x11_p_t[i][0].max()
                x_t[i][0][22] = x12_c_t[i][0].max()
                x_t[i][0][23] = x12_p_t[i][0].max()
                x_t[i][0][24] = x13_c_t[i][0].max()
                x_t[i][0][25] = x13_p_t[i][0].max()
                x_t[i][0][26] = x14_c_t[i][0].max()
                x_t[i][0][27] = x14_p_t[i][0].max()
                x_t[i][0][28] = x15_c_t[i][0].max()
                x_t[i][0][29] = x15_p_t[i][0].max()
                x_t[i][0][30] = x16_c_t[i][0].max()
                x_t[i][0][31] = x16_p_t[i][0].max()
                x_t[i][0][32] = x17_c_t[i][0].max()
                x_t[i][0][33] = x17_p_t[i][0].max()
                x_t[i][0][34] = x18_c_t[i][0].max()
                x_t[i][0][35] = x18_p_t[i][0].max()
                x_t[i][0][36] = x19_c_t[i][0].max()
                x_t[i][0][37] = x19_p_t[i][0].max()
                x_t[i][0][38] = x20_c_t[i][0].max()
                x_t[i][0][39] = x20_p_t[i][0].max()
                x_t[i][0][40] = x21_c_t[i][0].max()
                x_t[i][0][41] = x21_p_t[i][0].max()
                x_t[i][0][42] = x22_c_t[i][0].max()
                x_t[i][0][43] = x22_p_t[i][0].max()
                x_t[i][0][44] = x23_c_t[i][0].max()
                x_t[i][0][45] = x23_p_t[i][0].max()
                x_t[i][0][46] = x24_c_t[i][0].max()
                x_t[i][0][47] = x24_p_t[i][0].max()
                x_t[i][0][48] = x25_c_t[i][0].max()
                x_t[i][0][49] = x25_p_t[i][0].max()
                x_t[i][0][50] = x26_c_t[i][0].max()
                x_t[i][0][51] = x26_p_t[i][0].max()
                x_t[i][0][52] = x27_c_t[i][0].max()
                x_t[i][0][53] = x27_p_t[i][0].max()
                x_t[i][0][54] = x28_c_t[i][0].max()
                x_t[i][0][55] = x28_p_t[i][0].max()
                x_t[i][0][56] = x29_c_t[i][0].max()
                x_t[i][0][57] = x29_p_t[i][0].max()
                x_t[i][0][58] = x30_c_t[i][0].max()
                x_t[i][0][59] = x30_p_t[i][0].max()
                x_t[i][0][60] = x31_c_t[i][0].max()
                x_t[i][0][61] = x31_p_t[i][0].max()
                x_t[i][0][62] = x32_c_t[i][0].max()
                x_t[i][0][63] = x32_p_t[i][0].max()
                x_t[i][0][64] = x33_c_t[i][0].max()
                x_t[i][0][65] = x33_p_t[i][0].max()
                x_t[i][0][66] = x34_c_t[i][0].max()
                x_t[i][0][67] = x34_p_t[i][0].max()
                x_t[i][0][68] = x35_c_t[i][0].max()
                x_t[i][0][69] = x35_p_t[i][0].max()
                x_t[i][0][70] = x36_c_t[i][0].max()
                x_t[i][0][71] = x36_p_t[i][0].max()
                x_t[i][0][72] = x37_c_t[i][0].max()
                x_t[i][0][73] = x37_p_t[i][0].max()
                x_t[i][0][74] = x38_c_t[i][0].max()
                x_t[i][0][75] = x38_p_t[i][0].max()
                x_t[i][0][76] = x39_c_t[i][0].max()
                x_t[i][0][77] = x39_p_t[i][0].max()
                x_t[i][0][78] = x40_c_t[i][0].max()
                x_t[i][0][79] = x40_p_t[i][0].max()
                x_t[i][0][80] = x41_c_t[i][0].max()
                x_t[i][0][81] = x41_p_t[i][0].max()
                x_t[i][0][82] = x42_c_t[i][0].max()
                x_t[i][0][83] = x42_p_t[i][0].max()
                x_t[i][0][84] = x43_c_t[i][0].max()
                x_t[i][0][85] = x43_p_t[i][0].max()
                x_t[i][0][86] = x44_c_t[i][0].max()
                x_t[i][0][87] = x44_p_t[i][0].max()
                x_t[i][0][88] = x45_c_t[i][0].max()
                x_t[i][0][89] = x45_p_t[i][0].max()
                x_t[i][0][90] = x46_c_t[i][0].max()
                x_t[i][0][91] = x46_p_t[i][0].max()
                x_t[i][0][92] = x47_c_t[i][0].max()
                x_t[i][0][93] = x47_p_t[i][0].max()
                x_t[i][0][94] = x48_c_t[i][0].max()
                x_t[i][0][95] = x48_p_t[i][0].max()
                x_t[i][0][96] = x49_c_t[i][0].max()
                x_t[i][0][97] = x49_p_t[i][0].max()
                x_t[i][0][98] = x50_c_t[i][0].max()
                x_t[i][0][99] = x50_p_t[i][0].max()


            bn_r = bn_r.view(-1, 1, 1)
            # reward normalization
            # reward = (bn_r - bn_r.mean()) / (bn_r.std() + 1e-7) #这个e-7是防止溢出
            # kkw
            # reward = (bn_r - bn_r.mean()) / (bn_r.std())
            # no normalization
            # reward = bn_r
            with torch.no_grad():
                # target_v = reward + gamma * x_t #v_  x_
                target_v = bn_r + gamma * x_t
            # print(target_v, v)
            # kww
            optimizer.zero_grad()
            loss = nn.MSELoss()(target_v, x)  # v  x1
            loss.backward()
            optimizer.step()
            if (i_episode + 1) % 100 == 0:  # 原来是2000
                if (i_episode + 1) % 2000 == 0:  # 原来是2000，1000的结果和2000一样，不能是4000
                    target_net.load_state_dict(act_net.state_dict())
                avg_ep_reward = np.mean(ep_rewards)
                avg_ep_se = np.mean(ep_se)
                avg_ep_qoe = np.mean(ep_qoe)
                avg_ep_ssl = np.mean(ep_ssl)
                avg_es = np.mean(ep_es)
                avg_us = np.mean(ep_us)
                print('step:', (i_episode + 1), 'avg_reward:', avg_ep_reward, 'avg_se:', avg_ep_se, 'avg_qoe:',
                      avg_ep_qoe, 'avg_ssl:', avg_ep_ssl, 'embb', avg_es, 'urllc', avg_us)
                ep_rewards = []
                ep_se = []
                ep_qoe = []
                ep_ssl = []
                ep_es = []
                ep_us = []


    wave = np.array(wave)
    np.savetxt('50_DDQN.csv', wave, fmt="%.6f", delimiter=',')
    torch.save(act_net.state_dict(), '../models/act_ddqn_50')
    torch.save(target_net.state_dict(), '../models/target_ddqn_50')
    return avg_ep_latency, s, avg_ep_ssl

if __name__ == '__main__':
    sla = [0.5] *num
    target_net, act_net = Net(), Net()
    # act_net.load_state_dict(torch.load('../models/actNature'))
    # target_net.load_state_dict(torch.load('../models/targetNature'))
    l, s,ssl = rlRan(sla, target_net, act_net, 2000)

    print(l, s)