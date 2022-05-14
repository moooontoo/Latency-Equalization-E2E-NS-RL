# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:25:44 2019

@author: zheng
"""

import numpy as np
import copy
import time
from queue import Queue
import torch
import os
import sys
import pandas as pd
# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)

from Read_helpers import read_SN_VN, get_CostRatio_UtilizationRate
from ActiveSearch import active_search
from Embedding_and_Release import get_cost_matrix, update_SN, print_mapping_solution
from PerformanceEvaluation import *
from PtrNet import Ptr_Net, weights_init

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

##--------------------------------------------------------------------------
# VN_Link:dic保存所有虚拟网络的v_link key=VN的编号(下同)
# VN_Node:dic保存所有虚拟网络的v_node
# VN_Life:dic保存所有虚拟网络的编号,生存时间,结束时间(开始为空),失败次数 key=时序(0)
#        key时刻下需要映射的VN
##--------------------------------------------------------------------------

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print('Save model in {} successfully\n'.format(path))


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location= 'cpu'))
        print('Load model in {} successfully\n'.format(path))
    else:
        print('Cannot find {}'.format(path))
    return model


def experiment(hlp, time_step=3000, first_try=300,
               new_try=30, time_iter=300, batch_size=10,
               max_request_num=-1, use_node_utilization=False,
               s_input_information={}, use_neighbour_link_penalty=False,
               dropout=0, data_path='../data/data_500', load_model_path='../models/1', save_model_path='../models/1',device='cpu', if_print=False
               ):
    hlp = hlp
    file1 = os.path.join(data_path, 'maprecord.txt')
    #file2 = os.path.join(data_path, 'virtualnetworkTP.txt')
    file2 = os.path.join(data_path, 'virtualnetworkTP_40.txt')#验证无生存期限制的SFC映射，只有40个
    solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life = read_SN_VN(file1, file2)

    #file3 = os.path.join(data_path, 'VNETPDCresults.txt')
    # CostRatio, UtilizationRate = get_CostRatio_UtilizationRate(file3)

    ptr_net = Ptr_Net(hidden_size=128, batch_size=10, embedding_size=128, dropout_p=dropout,
                      s_input_information=s_input_information,
                      use_neighbour_link_penalty=use_neighbour_link_penalty,
                      device=device)
    if load_model_path != '':
        ptr_net = load_model(ptr_net, load_model_path)
    else:
        ptr_net.apply(weights_init)

    ptr_net.apply(weights_init)
    ptr_net.to(device=device)

    period_RC = dict()  # 收益成本比 key=时序 下同
    period_node_UT = dict()  # 资源利用率
    node_ut = 0
    Original_S_Links = copy.deepcopy(SN_Link)
    Original_S_Nodes = copy.deepcopy(SN_Node)
    s_links = copy.deepcopy(Original_S_Links)
    s_nodes = copy.deepcopy(Original_S_Nodes)

    periods_cost_matrixs = dict()

    longterm_revenue = 0.0
    longterm_cost = 0.0
    left_queue = Queue()  # 剩余的网络请求队列
    failed_queue = Queue()  # 上一个时刻失败的网络请求队列，会在下一个时刻放到剩余队列的尾部
    success_num = 0
    finish_time = 0
    total_num = 0

    upline_node_resources = 0
    upline_link_resources = 0

    # 只取300个虚拟网络请求，用于快速测试
    if max_request_num == -1:
        max_request_num = len(VN_Life[0])
    for vn_life in VN_Life[0]:
        if max_request_num == 0:
            break
        left_queue.put(vn_life)
        max_request_num -= 1
        id = vn_life[0]
        upline_node_resources += get_total_node_resources(VN_Node[id])
        upline_link_resources += get_total_link_resources(VN_Link[id])
    upline_rc = (upline_node_resources + upline_link_resources)/(upline_node_resources + upline_link_resources*2)
    upline_rc = '{:.3f}'.format(upline_rc)
    for t in range(0, time_step, 50):
        while not failed_queue.empty():
            left_queue.put(failed_queue.get())

        if not left_queue.empty():
            print('\033[1;31;40m这是在', t, '时刻下的虚拟网络映射请求,当前还剩下', left_queue.qsize(), '个虚拟网络请求等待映射\033[0m')
            print('释放前的物理网络node资源为', get_total_node_resources(s_nodes), 'link资源为', get_total_link_resources(s_links))

        if t in periods_cost_matrixs:
            if not left_queue.empty():
                print('当前时刻要释放', periods_cost_matrixs[t]['num_to_be_freed'], '个虚拟网络')
            update_SN(
                s_nodes=s_nodes,
                s_links=s_links,
                snode_update_matrix=+1 * periods_cost_matrixs[t]['snode_cost_matrix'],
                slink_update_matrix=+1 * periods_cost_matrixs[t]['slink_cost_matrix'],
            )

        if not left_queue.empty():
            print('释放后的物理网络node资源为', get_total_node_resources(s_nodes), 'link资源为', get_total_link_resources(s_links), '\n')

        if left_queue.empty():
            continue

        if t == 0:
            try_num = first_try
        else:
            if t in periods_cost_matrixs:
                try_num = new_try * periods_cost_matrixs[t]['num_to_be_freed']
            else:
                try_num = new_try
        try_num = min(left_queue.qsize(),try_num)
        R_C = []  # 收益成本比
        node_UT = []  # 节点资源利用率
        link_UT = []  # 链路资源利用率

        #记录每条SFC hop count
        sfc_hop = []

        # 每个时刻尝试映射left_queue中的try_num虚拟网络请求
        for i in range(try_num):  # 尝试把网络充满 默认尝试300次
            sys.stdout.write('\r{}/{}'.format(i,try_num))
            sys.stdout.flush()
            start_time = time.time()

            if left_queue.empty():
                break

            vn = left_queue.get()
            id = vn[0]
            life_time = vn[1]
            v_links = VN_Link[id]
            v_nodes = VN_Node[id]

            if if_print:
                print('\t\033[1;31;40m当前{}时刻,已映射{}个虚拟网络，当前尝试映射的第{}号虚拟网络\033[0m'.format(t, success_num, id))
                print('\t\t            v_node_num = ', len(v_nodes), 'v_link_num = ', len(v_links), 'v_life_time = ',
                      life_time)

            # active search 若能返回一个映射方案，那么这个映射方案必定包括成功的节点映射方案和成功的链路映射方案。否则就是不能返回一个能用的映射方案
            result = active_search(hlp[i],
                ptr_net=ptr_net,
                s_nodes=s_nodes,
                v_nodes=v_nodes,
                s_links=s_links,
                v_links=v_links,
                batch_size=batch_size,
                iter_time=time_iter,
                current_node_utilization=node_ut,
                use_node_utilization=use_node_utilization,
                device=device
                )
            total_num += 1

            ptr_net = result['ptr_net']
            embedding_success = result['best_mapping_solution']['embedding_success']

            # embedding_success为false时，表示无法找到一个完整的成功的映射方案（包括成功的节点映射和链路映射方案）
            if not embedding_success:
                if if_print:
                    print('\t\t此虚拟网络映射失败')  # 节点映射失败或链路映射失败
                failed_queue.put(vn)
            else:
                if if_print:
                    print('\t\t此虚拟网络映射成功')
                    print_mapping_solution(result['best_mapping_solution'])

                success_num += 1
                cost_matrix = get_cost_matrix(
                    snode_num=len(s_nodes),
                    v_nodes=v_nodes,
                    mapping_solution=result['best_mapping_solution']
                )

                added_revenue = get_total_resources(nodes=v_nodes, links=v_links)
                added_node_cost = cost_matrix['snode_cost_matrix'].sum()
                added_link_cost = cost_matrix['slink_cost_matrix'].sum() / 2.0
                added_total_cost = added_node_cost + added_link_cost
                longterm_revenue += added_revenue
                longterm_cost += added_total_cost

                if if_print:
                    print(
                        '\n\t\t映射此虚拟网络请求获得收益为{},节点资源损耗为{},链路资源损耗为{},总资源损耗为{}'.format(
                            added_revenue,
                            added_node_cost,
                            added_link_cost,
                            added_total_cost
                        )
                    )
                    print(
                        '\t\t映射此虚拟网络前物理网络节点资源剩余 {},链路资源剩余 {}'.format(
                            get_total_node_resources(s_nodes),
                            get_total_link_resources(s_links)
                        )
                    )

                # 使用更新矩阵 更新物理网络
                s_nodes, s_links = update_SN(
                    s_nodes=s_nodes,
                    s_links=s_links,
                    snode_update_matrix=-1 * cost_matrix['snode_cost_matrix'],
                    slink_update_matrix=-1 * cost_matrix['slink_cost_matrix'],
                )

                if if_print:
                    print(
                        '\t\t映射此虚拟网络后物理网络节点资源剩余 {},链路资源剩余 {}'.format(
                            get_total_node_resources(s_nodes),
                            get_total_link_resources(s_links)
                        )
                    )

                if (t + life_time) not in periods_cost_matrixs:
                    periods_cost_matrixs.update({
                        (t + life_time): {
                            'num_to_be_freed': 0,
                            'snode_cost_matrix': np.zeros(shape=(len(s_nodes), 1), dtype=float),
                            'slink_cost_matrix': np.zeros(shape=(len(s_nodes), len(s_nodes)), dtype=float),
                        }
                    })
                periods_cost_matrixs[t + life_time]['num_to_be_freed'] += 1
                periods_cost_matrixs[t + life_time]['snode_cost_matrix'] += cost_matrix['snode_cost_matrix']
                periods_cost_matrixs[t + life_time]['slink_cost_matrix'] += cost_matrix['slink_cost_matrix']

            #计算一条SFC映射成功后的跳数
            sfc = 0
            for value in result['best_mapping_solution']['link_mapping_solution'].values():
                sfc += len(value) -1
            sfc_hop.append(sfc)

            # 计算rc和utilization
            longterm_rc = get_revenue_cost_ratio(longterm_revenue, longterm_cost)
            node_ut = get_node_utilization(current_s_nodes=s_nodes, original_s_nodes=Original_S_Nodes)
            link_ut = get_link_utilization(current_s_links=s_links, original_s_links=Original_S_Links)
            finish_time = t
            R_C.append('{:.3f}'.format(longterm_rc))
            node_UT.append('{:.3f}'.format(node_ut))
            link_UT.append('{:.3f}'.format(link_ut))

            if if_print or i == (try_num-1):
                print(
                    '\n\t\tlong-term revenue = {}, long-term cost = {}, long-term r_c = {:.3f}, node utilization = {:.4f}, link utilization = {:.4f}'.format(
                        longterm_revenue,
                        longterm_cost,
                        longterm_rc,
                        node_ut,
                        link_ut
                    )
                )
                print('\t\t运行时间为', time.time() - start_time, 's\n')


        if not R_C:
            longterm_rc = get_revenue_cost_ratio(longterm_revenue, longterm_cost)
            node_ut = get_node_utilization(current_s_nodes=s_nodes, original_s_nodes=Original_S_Nodes)
            link_ut = get_link_utilization(current_s_links=s_links, original_s_links=Original_S_Links)
            finish_time = t
            R_C.append('{:.3f}'.format(longterm_rc))
            node_UT.append('{:.3f}'.format(node_ut))
            link_UT.append('{:.3f}'.format(link_ut))

        period_RC.update({(t): R_C})
        period_node_UT.update({(t): node_UT})
        if save_model_path != '':
            save_model(ptr_net, save_model_path)

    print('已结束并完成了存储，中间数据存储在logs文件夹中')

    return finish_time, period_RC, period_node_UT, 1.0 * success_num / total_num, upline_rc, sfc_hop


def test(hlp, data_path='../data/data_500', batch_size=10, iteration_num=10, dropout=0, load_model_path='../models/1',
         save_model_path='../models/1', log_path='../logs', device='cpu', input_type=2, max_request_num=-1,first_try=250):
    '''
        完全测试参数： max_request_num=500, first_try = 250
        简单快速测试： max_request_num=50, first_try = 50
    '''
    batch_size = batch_size
    time_iter = iteration_num
    time_step = 1500
    first_try = first_try
    hlp = hlp
    use_node_utilization = False
    dropout = dropout
    s_input_information = {
        'snode index': False,  # 仅使用每个物理节点下标作为网络输入
        'snode resource': False,  # 仅使用每个物理节点资源值作为网络输入
        'snode resource and neighbour link resource': False  # 使用 1.每个物理节点资源值
        #      2. 每个物理节点直接相邻的剩余链路资源总和值
        #      3. 每个物理节点直接相邻的剩余的最小链路边的资源值 作为网络输入
    }
    if input_type == 1:
        choosed_information = 'snode index'
    if input_type == 2:
        choosed_information = 'snode resource'
    if input_type == 3:
        choosed_information = 'snode resource and neighbour link resource'
    s_input_information[choosed_information] = True

    # 惩罚某些不符合要求的物理节点，使算法选择该物理节点的概率大大降低
    # 对已经映射过同一虚拟网络其他虚拟节点的物理节点惩罚，即一个虚拟网络的虚拟节点不会映射到一个相同的物理节点上
    use_already_played_penalty = True

    # 对物理节点资源小于虚拟节点资源的物理节点做惩罚
    use_node_resource_penalty = True

    # 1. 对物理节点剩余的直接相邻的链路资源总和值 小于 虚拟节点直接相邻的链路资源总和值 的物理节点做惩罚
    # 2. 对物理节点剩余的直接相邻的链路资源最大值 小于 虚拟节点直接相邻的链路资源最小值 的物理节点做惩罚
    use_neighbour_link_penalty = True

    finish_time, period_RC, period_node_UT, success_ratio, upline_rc, sfc_hop = experiment(
        hlp,
        time_step=time_step,
        first_try=first_try,  # 第一次约250全满，若只用于测试10就行
        new_try=5,
        time_iter=time_iter,
        batch_size=batch_size,
        max_request_num=max_request_num,
        use_node_utilization=use_node_utilization,
        s_input_information=s_input_information,
        use_neighbour_link_penalty=use_neighbour_link_penalty,
        dropout=dropout,
        load_model_path=load_model_path,
        save_model_path=save_model_path,
        data_path=data_path,
        device=device
    )


    print('\n\n\nbatch_size = {}, iter = {}, max_request_num = {}, dropout = {}'.format(batch_size, time_iter, max_request_num, dropout))
    print('神经网络的输入信息为{}'.format(choosed_information))
    print('在{}时刻完成{}个虚拟网络的映射,映射的成功率 = {},最终长期RC值 = {}, 理论最优RC值 = {}\n\n'.format(
            finish_time, max_request_num,
            success_ratio, period_RC[finish_time][-1],
            upline_rc
        )
    )

    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    df = pd.DataFrame()
    if os.path.exists('../logs/result.xlsx'):
        df = pd.read_excel('../logs/result.xlsx',engine='openpyxl')
    else:
        df = pd.DataFrame(
            columns=("time","batch size", "iteration", "max request",
                     "acceptance ratio","longterm rc","upline rc",
                     "input","dropout","use pretrained model")
        )
    df = df.append({
            "time": local_time,
            "batch size": batch_size,
            "iteration": iteration_num,
            "max request": max_request_num,
            "acceptance ratio": success_ratio,
            "longterm rc": float(period_RC[finish_time][-1]),
            "upline rc": float(upline_rc),
            "input": choosed_information,
            "dropout": dropout,
            "use pretrained model": load_model_path if load_model_path else 'False'
        },
        ignore_index=True
    )
    df = df.sort_values(by=['max request','batch size','iteration','longterm rc'], ascending=False)     #按这三列  降序排列
    df.to_excel('../logs/result.xlsx', index=0)      #index : bool, default True        Write row names (index).


    df = pd.DataFrame(
        data={
            'time':[item[0] for item in period_RC.items()],
            'longterm rc':[item[1][-1] for item in period_RC.items()],
            'ut':[item[1][-1] for item in period_node_UT.items()]
        }
    )

    print(type(local_time))
    local_time = local_time.replace(' ','_')

    print(local_time)
    local_time = local_time.encode(encoding='utf8')
    print(local_time)
    df.to_excel('../logs/{}.xlsx'.format(local_time,encoding='utf8'), index=0)
    df.to_excel('../logs/123.xlsx', index=0)

    return sfc_hop
