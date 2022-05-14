# coding=utf-8
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
from config import *
from utils import *
import random

import matplotlib.pyplot as plt
import torch


class Network:

    def __init__(self, config, show_config=True):  # 尽量搞懂这些是啥意思
        self.config = config
        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)  # 把NetworkConfig里面的变量和值   当成字典输出

        if show_config:
            print('Network configs...')
            pp(self._attrs)

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

        self.start()

    def start(self):
        # 用户基站距离
        self.mbs_u_distance = []
        self.sbs_u_distance = []

        # 用户基站位置
        self.mbs_location = [45, 55]
        self.sbs_location = [55, 45]
        # self.u_location = [[10., 55.], [25., 35.], [35., 70.], [60., 25.], [75., 45.], [80., 60.], [6., 23.],
        #                    [46., 21.], [12., 76.], [53., 50.], [75., 57.], [93., 65.], [10., 78.], [91., 51.],
        #                    [67., 91.], [15., 90.], [23., 34.], [49., 79.], [75., 14.], [12., 69.]]
        self.u_location = [ [10., 55.], [25., 35.], [35., 70.], [60., 25.], [75., 45.], [80., 60.], [6., 23.],
                           [46., 21.], [12., 76.], [53., 50.], [75., 57.], [93., 65.], [10., 78.], [91., 51.],
                           [67., 91.], [15., 90.], [23., 34.], [49., 79.], [75., 14.], [12., 69.],
                           [55,10],[35,25],[70,35],[25,60],[45,75],[60,80],[23,6],[21,46],[76,12],[50,53],
                           [57.,75.],[65.,93.],[78.,10.],[51.,91.],[91.,67],[90.,15.],[34.,23.],[79.,49.],[14.,75.],[69.,12.] ]

        self.create_topology()

    # 得到用户和基站的信道增益gain
    def create_topology(self):
        all_location = np.vstack((self.mbs_location, self.sbs_location, self.u_location))
        all_distance = pdist(all_location, 'euclidean')

        self.mbs_u_distance = all_distance[1:1 + self.u_number]
        self.sbs_u_distance = all_distance[1 + self.u_number:1 + 2 * self.u_number]

        self.gain_sbs_u = self._channel_gain(self.sbs_u_distance, shape=self.u_number)  # 6
        self.gain_mbs_u = self._channel_gain(self.mbs_u_distance, shape=self.u_number)  # 6

    # 计算信道增益
    def _channel_gain(self, distance, shape):
        path_loss_bs_user = 37 + 30 * np.log2(distance)
        path_loss_bs_user = path_loss_bs_user + self.generate_shadow_fading(0, 8, shape, 1)
        gain = np.power(10, -path_loss_bs_user / 10)
        return gain

    def generate_shadow_fading(self, mean, sigma, num_user, num_bs):
        '''
            本函数生成对数正态分布的阴影衰落
            :param mean:  均值dB
            :param sigma: 标准差 dB
            :param num_bs:
            :param num_user:
            :return:
        '''
        sigma = np.power(10, sigma / 10)
        mean = np.power(10, mean / 10)
        m = np.log(mean ** 2 / np.sqrt(sigma ** 2 + mean ** 2))
        sigma = np.sqrt(np.log(sigma ** 2 / mean ** 2 + 1))
        lognormal_fade = np.exp(np.random.randn(num_user, num_bs) * sigma + m)
        return lognormal_fade.reshape(num_user)

    def draw_net(self):
        '''画出PU SU分布图保存为model.jpg'''
        # # plt.subplot(111)
        # pt_location = self.pt_location
        # pr_location = self.pr_location
        # st_location = self.st_location
        # sr_location = self.sr_location

        # plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
        # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        # plt.plot(pt_location[:, 0], pt_location[:, 1], '+', color='red', ms=15, label='PT')
        # plt.plot(pr_location[:, 0], pr_location[:, 1], '.r', ms=15, label='PR')
        # plt.plot(st_location[:, 0], st_location[:, 1], 's', color='blue', ms=10, label='ST')
        # plt.plot(sr_location[:, 0], sr_location[:, 1], '^b', ms=8, label='SR')
        # # plt.plot(su_location[:, 0], su_location[:, 1], '^b', ms=8, label='SU')
        # # plt.plot(30, 70, '+', color='red', ms=15, label='PBS')
        # # plt.plot(70, 30, 's', color='blue', ms=10, label='CBS')
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        # plt.legend()
        # plt.savefig('model.jpg')
        # plt.show()

    '''def pu_snr(self, su_p, su_c):
        p_num = self.config.pu_number
        s_num = self.config.su_number
        pu_p = self.pu_p
        pu_c = self.pu_c
        pus_snr = []
        pus_rate = []
        for i in range(p_num):
            pu_signal = 0.
            su_signal = 0.
            for j in range(self.config.c_num):
                if pu_c[i][j] == 1:
                    pu_signal += pu_p[i][j] * self.gain_pbs_pu[i]
                    for k in range(s_num):
                        su_signal += su_p[k][j] * su_c[k][j] * self.gain_mbs_u[k]
            interference = su_signal + self.config.N0
            pu_snr = pu_signal / interference
            pu_rate = self.BW_sc*math.log2(1 + pu_snr) #假设带宽为1
            pus_snr.append(pu_snr)
            pus_rate.append(pu_rate)
        return pus_snr, pus_rate
        '''

    def u_rate(self, u_p, u_c):
        u_num = self.config.u_number
        us_rate = []
        for i in range(u_num):
            u_rate = 0.
            u_interference = 0.
            for j in range(self.config.c_num):
                if u_c[i][j] == 1.:
                    u_signal = u_p[i][j] * self.gain_mbs_u[i]
                    for i_ in range(u_num):
                        if (u_c[i_][j] == 1.) and (i != i_):
                            u_interference += u_p[i_][j] * self.gain_mbs_u[i_]
                            # 同道干扰
                    interference = u_interference + self.config.N0
                    # interference = self.config.N0
                    u_snr = u_signal / interference
                    u_rate = self.BW_sc * math.log2(1 + u_snr)
            us_rate.append(u_rate)
        return us_rate

    '''
        def compute_reward(self, u_p, u_c):
            # 网络频谱效率计算
            sus_rate = self.u_rate(u_p, u_c)
            sus_rate_sum = np.sum(sus_rate)
            #频谱效率
            n_se = sus_rate_sum/(self.c_num*self.BW_sc)
            # 感知用户QoE计算
            latency = np.zeros(20)
            latency_limit = 10000
            rate_limit = 0.5e3
            Qoe_counter = 0
            arrive_package_rate = np.random.poisson(lam=1,size=10)
            #对两类用户，判断满足QoE的用户数量
            for i in range(self.u_number):
                if i < 10:
                    # Qoe_counter+=1
                    if sus_rate[i] >= rate_limit:
                        Qoe_counter+=1
                else:
                    latency[i] = 1./(sus_rate[i] - arrive_package_rate[i-10]+1e-7)
                    if latency[i] <= latency_limit:
                        Qoe_counter+=1

            QoE = Qoe_counter/self.u_number
            # SE与QoE线性组合
            reward_alpha = 0
            reward = reward_alpha*n_se + (1-reward_alpha)*QoE
            reward_c = reward

            return reward_c,n_se,QoE
    '''

    # 提供初始状态
    def state_first(self):
        # u_c = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        # u_c = [[1.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,
        #        [0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,[0.]*30,
        #        [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30,
        #        [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30, [0.] * 30]
        u_c = []
        for i in range(self.u_number):
            if i == 0:
                u_c.append([1.]*self.u_number)
            else:
                u_c.append([0.]*self.u_number)

        u_c = np.array(u_c)
        u_p =[]
        for i in range(self.u_number):
            a = [0.]*self.u_number
            for j in range(self.u_number):
                if j == i:
                    a[j] = 1.
                else:
                    a[j] = 0.
            u_p.append(a)
        # u_p = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
        u_p = np.array(u_p)
        # pus_snr, _ = self.pu_snr(su_p, su_c) # 获得主用户SINR

        # 各用户的数据率
        us_rate = self.u_rate(u_c, u_p)

        return us_rate

    '''用于计算每个用户空口时延 0704'''

    def u_latency(self, u_p, u_c, sla):
        # 用户空口时延计算
        latency = np.zeros(self.u_number)
        # arrive_package_rate = np.random.poisson(lam=100, size=20)
        # 对比
        arrive_package_rate = np.random.poisson(lam=1, size=self.u_number)
        sus_rate = self.u_rate(u_p, u_c)
        for i in range(self.u_number):
            if i < self.sep:
                # latency[i] = 4e7 / (sus_rate[i] - 2e5 + 1e-7)
                # latency[i] = 20 + 5.7e7 / (sus_rate[i] + 1e-7)  #0812
                latency[i] = 50 + 5.7e7 / (sus_rate[i] + 1e-7)
                # latency[i] = 50 + 3.6e7 / (sus_rate[i] + 1e-7)   #0814
                # latency[i] = 20 + (sus_rate[i] + 1e-7)/self.rate_limit * 10
                if sus_rate[i] < self.rate_limit or latency[i] > self.latency_limit2:
                    latency[i] = self.latency_limit2
            else:
                # latency[i] = 10. / (sus_rate[i] - arrive_package_rate[i] + 1e-7)
                # 对比算法
                latency[i] = 1. / (sus_rate[i] - arrive_package_rate[i] + 1e-7)
                if latency[i] < 0 or latency[i] > self.latency_limit:
                    latency[i] = self.latency_limit
        return latency

    '''单纯根据不同切片空口时延比例的计算QoE  作为reward，没加入频谱效率的权重 0705'''

    def compute_reward(self, u_p, u_c, sla):
        # 网络频谱效率计算
        sus_rate = self.u_rate(u_p, u_c)
        sus_rate_sum = np.sum(sus_rate)
        # sla = [0.9] * self.u_number
        latency = self.u_latency(u_p, u_c, sla)
        # 频谱效率
        n_se = sus_rate_sum / (self.c_num * self.BW_sc)
        # 用户QoE计算
        Qoe_counter = 0
        # embb用户满意度
        # ssl_1, ssl_2 = np.zeros(self.sep), np.zeros(self.sep)
        ssl_1 = sus_rate[0: self.sep]
        ssl_1 = np.array(ssl_1) / self.rate_limit + 1
        # ssl_2 = np.log2(ssl_1)
        # ssl_2 = np.log(ssl_1)
        ssl_2 = np.log10(ssl_1)
        ssl = np.sum(ssl_2)
        # for i in range(self.sep):
        #     ssl = ssl * ssl_2[i]

        # 对两类用户，判断满足QoE的用户数量
        # for i in range(self.u_number):
        #     if i < 10:
        #         # Qoe_counter+=1
        #         if sus_rate[i] >= self.rate_limit:
        #             Qoe_counter += 1
        #     elif latency[i] <= self.latency_limit * sla[i]:
        #             Qoe_counter += 1

        # 都转换成时延的约束
        for i in range(self.u_number):
            if i < self.sep:
                if latency[i] < self.latency_limit2 * sla[i]:
                    Qoe_counter += 1
            elif i == self.sep:
                es = Qoe_counter
                if latency[i] < self.latency_limit * sla[i]:
                    Qoe_counter += 1
            elif latency[i] < self.latency_limit * sla[i]:
                Qoe_counter += 1

        us = Qoe_counter - es
        QoE = Qoe_counter / self.u_number
        # SE与QoE线性组合
        reward_alpha = self.reward_alpha
        # reward = reward_alpha * n_se + (1 - reward_alpha) * QoE
        reward = reward_alpha * ssl + (1 - reward_alpha) * QoE
        return reward, n_se, QoE, latency, ssl, es, us
