#from ran import *
from users_plus.DDQN_50 import rlRan
from users_plus.DeepQNetwork_50 import *
from Test_50 import test
import copy
import os
import numpy as np
np.random.seed(0)

from Plot import plot
lowerbound = 0.1
upperbound = 0.9

cs1 = 200  #eMBB latency limit
cs2 = 10   #URLLC l..  l..
num = 50  # total users
sep = 25  # embb users' number
alpha = 0
rate_limit = 3e5
train = 200

if __name__ == '__main__':
    #各切片E2E时延约束集合
    t = [cs1]*sep + [cs2]*(num-sep)
    t = np.array(t)

    #随机或者固定比例 初始化.  sla代表端到端时延约束分配给ran的比例
    #sla = [0.6]*10 + [0.4]*10
    # sla = [0.75]*sep + [0.25]*(20-sep)
    sla = [0.8]*sep + [0.2]*(num-sep)

    sla = np.array(sla)
    #sla = np.random.rand(20)
    #sla = np.random.randint(1,10,20)/10
    old = copy.deepcopy(sla)
    SLA = []
    SLA.append(copy.deepcopy(sla))
    print('***********第1次**********************第1次**********************第1次***********')
    target_net, act_net = Net(), Net()
    # act_net.load_state_dict(torch.load('../models/act'))
    # target_net.load_state_dict(torch.load('../models/target'))
    l1, r, ssl =rlRan(sla, target_net, act_net, train)   #0.001的学习率，300个episode得到一个接近最终的结果
    #np.savetxt('r.csv', r, fmt="%.6f", delimiter=',')

    e_ran, e_cn = 0, 0  # embb ran/cn accessed
    u_ran, u_cn = 0, 0  # urllc ran/cn accessed

    sla2 = 1 - sla #端到端时延约束分配给cn的比例
    s_h = 2
    hlp = s_h * np.multiply(t, sla2) #时延转换成跳数
    sfc_hop = np.array(test(hlp))  #
    l2 = sfc_hop/ s_h
    l2[0:sep] += 20 #前sep个是embb用户，额外的时延

    d1 = np.multiply(t, sla) - l1  # RAN时延余量
    d2 = np.multiply(t, sla2) - l2  # cn时延余量
    credit = d1 + d2  #credit 若大于某值，表示部分切片存在时延余量，存在优化资源分配的空间

    xi = 50          #迭代划分比例的终止条件  试探得出
    xii = 5
    qoe = 0
    QoE = []
    Reward =[]
    iter = 0
    #每个切片SLA划分不同
    # while any(credit > xi):
    #thresh = min(sep * 40, 1200)
    thresh = 900
    while sum(credit) > thresh:
        qoe = 0
        e_ran, e_cn = 0, 0  # embb ran/cn accessed
        u_ran, u_cn = 0, 0  # urllc ran/cn accessed
        reward = 0
        # 计算成功连接的人数  函数的输出可以选择吗？
        for i in range(num):
            if d1[i] > 0 and d2[i] > 0:
                qoe += 1
            if i < sep:
                if d1[i] > 0:
                    e_ran += 1
                if d2[i] > 0:
                    e_cn += 1
            else:
                if d1[i] > 0:
                    u_ran += 1
                if d2[i] > 0:
                    u_cn += 1
        QoE.append(copy.deepcopy(qoe))
        reward = alpha * ssl + (1 - alpha) * qoe
        Reward.append(copy.deepcopy(reward))
        print('第', (iter + 1), '次迭代，E2E接入人数为：', qoe, 'SSL:', ssl, 'reward=', reward,
              'eMBB切片:RAN接入', e_ran, 'CN接入', e_cn, 'URLLC切片:RAN接入', u_ran, 'CN接入', u_cn)
        # np.savetxt('qoe__.csv', QoE, fmt="%.6f", delimiter=',')  # 保存为2位小数的浮点数，用逗号分隔
        for i in range(num):
            if i < sep:
                if credit[i] > xi:    #不满足约束的切片  进行片内均衡
                    if d1[i] > 0 and d2[i] < 0:
                        sla[i] = max(sla[i] - 0.1, lowerbound)
                    elif d2[i] > 0 and d1[i] < 0:
                        sla[i] = min(sla[i] + 0.1, upperbound)
                    elif d1[i] > d2[i]:     #满足约束的切片  进行片间均衡
                        sla[i] = max(sla[i] - 0.05, lowerbound)
                    elif d2[i] > d1[i]:
                        sla[i] = min(sla[i] + 0.05, upperbound)
            elif credit[i] > xii:
                if d1[i] > 0 and d2[i] < 0:
                    sla[i] = max(sla[i] - 0.1, lowerbound)
                elif d2[i] > 0 and d1[i] < 0:
                    sla[i] = min(sla[i] + 0.1, upperbound)
                elif d1[i] > d2[i]:  # 满足约束的切片  进行片间均衡
                    sla[i] = max(sla[i] - 0.05, lowerbound)
                elif d2[i] > d1[i]:
                    sla[i] = min(sla[i] + 0.05, upperbound)

        iter += 1
        if qoe == num or iter == 35 or (old == sla).all():
            break
        # if (old == sla).all():
        #     break
        old = copy.deepcopy(sla)
        SLA.append(copy.deepcopy(sla))
        print('***********第',(iter+1),'次***********','***********第',(iter+1),'次***********','***********第',(iter+1),'次***********')
        l1, r, ssl = rlRan(sla, target_net, act_net, train)
        np.savetxt('r.csv', r, fmt="%.6f", delimiter=',')

        sla2 = 1 - sla
        sfc_hop = np.array(test(hlp))
        l2 = sfc_hop / s_h
        l2[0:sep] += 20
        d1 = np.multiply(t, sla) - l1  # RAN时延余量
        d2 = np.multiply(t, sla2) - l2  # cn时延余量
        credit = d1 + d2# 也= t - l1 -l2
        # np.savetxt('d1.csv', d1, fmt="%.6f", delimiter=',')  # 保存为整数
        # np.savetxt('credit.csv', credit, fmt="%.6f", delimiter=',')  # 保存为整数
        # np.savetxt('sla.csv', SLA, fmt="%.6f", delimiter=',')  # 保存为整数

    SLA.append(copy.deepcopy(sla))
    qoe = 0
    # for i in range(20):
    #     if i < 10:
    #         if r[0][i] >= rate_limit:  # rate limit
    #             qoe += 1
    #     elif d1[i] > 0 and d2[i] > 0:
    #         qoe += 1
    #统一成时延约束后
    for i in range(num):
        if d1[i] > 0 and d2[i] > 0:
            qoe += 1
        if i < sep:
            if d1[i] > 0:
                e_ran += 1
            if d2[i] > 0:
                e_cn += 1
        else:
            if d1[i] > 0:
                u_ran += 1
            if d2[i] > 0:
                u_cn += 1
    QoE.append(copy.deepcopy(qoe))
    reward = alpha * ssl + (1 - alpha) * qoe
    Reward.append(copy.deepcopy(reward))

    plot(QoE, len(QoE), 'One slice one sla QoE changes during equalization', 0, len(QoE))

    #np.savetxt('ITERATIONS.csv', iter, fmt="%.6f", delimiter=',')  # 保存为整数
    # np.savetxt('d1.csv', d1, fmt="%.6f", delimiter=',')  # 保存为整数
    # np.savetxt('credit.csv', credit, fmt="%.6f", delimiter=',')  # 保存为整数
    # np.savetxt('sla.csv', SLA, fmt="%.6f", delimiter=',')  # 保存为整数
    # np.savetxt('qoe.csv', QoE, fmt="%.6f", delimiter=',')  # 保存为2位小数的浮点数，用逗号分隔
    # print('max E2E QoE=', max(QoE), "\nbest sla:", SLA[np.argmax(QoE)],'iteration:', (np.argmax(QoE)+1))
    print('max E2E QoE=', max(QoE), "\nbest sla:", SLA[np.argmax(QoE)], 'iteration:', (np.argmax(QoE) + 1), 'reward =',
          Reward[np.argmax(QoE)])
    print('max E2E Reward=', max(Reward), "\nbest sla:", SLA[np.argmax(Reward)], 'iteration:', (np.argmax(Reward) + 1),
          'QoE = ', QoE[np.argmax(Reward)])
