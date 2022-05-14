from ran import *
from DeepQNetwork import *
import copy
import os
np.random.seed(0)
from Test import test



if __name__ == '__main__':
    #各切片E2E时延约束集合
    t = [100]*10 + [10]*10
    t = np.array(t)

    #随机或者固定比例 初始化.  sla代表端到端时延约束分配给ran的比例
    sla =[0.1]*10 + [0.9]*10
    #sla = [float(np.random.rand(1))]*10 + [float(np.random.rand(1))]*10
    sla = np.array(sla)

    SLA.append(copy.deepcopy(sla))
    target_net, act_net = Net(), Net()
    act_net.load_state_dict(torch.load('../models/act'))
    target_net.load_state_dict(torch.load('../models/target'))
    l1, r =rlRan(sla, target_net, act_net, 300)   #0.001的学习率，300个episode得到一个接近最终的结果

    sla2 = 1 - sla #端到端时延约束分配给cn的比例
    s_h = 2
    hlp = s_h * np.multiply(t, sla2) #时延转换成跳数
    sfc_hop = np.array(test(hlp))  #
    l2 = sfc_hop/ s_h

    d1 = np.multiply(t, sla) - l1  # RAN时延余量
    d2 = np.multiply(t, sla2) - l2  # cn时延余量

    # 计算成功连接的人数  函数的输出可以选择吗？
    for i in range(20):
        if i < 10:
            if r[0][i] >= 1e4:  # rate limit
                qoe += 1
        elif d1[i] > 0 and d2[i] > 0:
            qoe += 1

    np.savetxt('sla_static.csv', sla, fmt="%.6f", delimiter=',')  # 保存为整数
    np.savetxt('qoe_static.csv', qoe, fmt="%.6f", delimiter=',')  # 保存为2位小数的浮点数，用逗号分隔