import numpy as np
import torch
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
#用到了节点数量
def short_path(slinks, start, end, v_bandwidth):  # 最短路径
    node_num = 148
    inf = 1000
    node_map = np.zeros(shape=(node_num, node_num), dtype=float)
    for link in slinks:
        node_map[link[0]][link[1]] = node_map[link[1]][link[0]] = link[2]

    hop = [inf for i in range(node_num)]
    visited = [False for i in range(node_num)]
    pre = [-1 for i in range(node_num)]
    hop[start] = 0


    for i in range(node_num):
        u = -1
        for j in range(node_num):
            if not visited[j]:
                if u == -1 or hop[j] < hop[u]:
                    u = j
        visited[u] = True
        if u == end:
            break

        for j in range(node_num):
            if hop[j] > hop[u] + 1 and node_map[u][j] >= v_bandwidth:
                hop[j] = hop[u] + 1
                pre[j] = u

    path = []
    if pre[end] !=-1:
        v = end
        while v!=-1:
            path.append(v)
            v = pre[v]
        path.reverse()

    return path

def link_embedding(s_links, slink_path, v_bandwidth):
    for i in range(1, len(slink_path)):
        u = slink_path[i - 1]
        v = slink_path[i]
        for j in range(len(s_links)):
            u2 = s_links[j][0]
            v2 = s_links[j][1]
            if (u == u2 and v == v2) or (u == v2 and v == u2):
                s_links[j][2] -= v_bandwidth


def link_release(s_links, slink_path, v_bandwidth):
    for i in range(1, len(slink_path)):
        u = slink_path[i - 1]
        v = slink_path[i]
        for j in range(len(s_links)):
            u2 = s_links[j][0]
            v2 = s_links[j][1]
            if (u == u2 and v == v2) or (u == v2 and v == u2):
                s_links[j][2] += v_bandwidth


def get_hops_and_link_consumptions(s_nodes, s_links, v_nodes, v_links, node_mapping):
    '''
    :param s_nodes: 物理节点资源 list (s_node_num,)
    :param s_links: 物理链路资源 list (s_link_num,), struct s_link = (u, v, bandwidth)
    :param v_nodes: 虚拟节点资源 list (v_node_num,)
    :param v_links: 虚拟链路资源 list (v_link_num,), struct v_link = (u, v, bandwidth)
    :param node_mapping: 节点映射方案
    :return: embedding success 是否映射成功; link_mapping_solutions 链路映射方案, link_consumptions 链路映射消耗, hops 链路映射消耗跳数
    '''

    batch_size = node_mapping.shape[0]
    v_node_num = node_mapping.shape[1]
    hops = torch.zeros(size=(batch_size, 1))
    link_consumptions = torch.zeros(size=(batch_size, 1))
    link_mapping_solutions = [dict() for i in range(batch_size)]
    embedding_success = [False for i in range(batch_size)]

    for i in range(batch_size):
        node_mapping_success = True
        link_mapping_success = True

        for j in range(v_node_num):
            if s_nodes[node_mapping[i][j]] < v_nodes[j]:
                node_mapping_success = False
                link_mapping_success = False
                # print('节点映射失败')
                break

        v_link_consumption_sum = 0
        for v_link in v_links:
            v_link_consumption_sum += v_link[2]

        if node_mapping_success:
            embedded_paths = []
            for v_link in v_links:
                v_from_node = v_link[0]
                v_to_node = v_link[1]
                v_bandwidth = v_link[2]

                s_from_node = node_mapping[i][v_from_node]
                s_to_node = node_mapping[i][v_to_node]

                s_path = short_path(s_links, s_from_node, s_to_node, v_bandwidth)

                if s_path == []:
                    link_mapping_success = False
                    # print('链路映射失败')
                    break
                else:
                    link_embedding(s_links, s_path, v_bandwidth)
                    embedded_paths.append([s_path, v_bandwidth])
                    hops[i][0] += len(s_path) - 1
                    #一条虚拟链路映射带来的带宽支出
                    link_consumptions[i][0] += (len(s_path) - 1) * v_bandwidth
                    link_mapping_solutions[i].update({v_link: s_path})

            for path, v_bandwidth in embedded_paths:
                link_release(s_links, path, v_bandwidth)

            if not link_mapping_success:
                hops[i] = 7 * len(v_links)
                link_consumptions[i] = v_link_consumption_sum * 7 * 2
                link_mapping_solutions[i] = dict()
        else:
            hops[i] = 7 * len(v_links)
            link_consumptions[i] = v_link_consumption_sum * 7 * 2
            link_mapping_solutions[i] = dict()

        if node_mapping_success and link_mapping_success:
            embedding_success[i] = True

    return embedding_success, link_mapping_solutions, link_consumptions, hops


# 给定一个映射方案mapping solution（包括节点映射方案node mapping solution和链路映射方案link mapping solution），
# 更新当前的物理网络资源s_nodes和s_links
def update_SN(s_nodes, s_links, snode_update_matrix, slink_update_matrix):  # 更新物理链路节点和带宽资源
    '''
        s_nodes : 当前物理网络节点资源
        s_links : 当前物理网络链路资源
        snode_update_matrix: 物理节点资源更新矩阵，映射网络时是负的节点资源cost矩阵，释放网络时是正的节点资源cost矩阵
        slink_update_matrix: 物理链路资源更新矩阵，映射网络时是负的链路资源cost矩阵，释放网络时是正的链路资源cost矩阵
        return: s_nodes,更新后的s_nodes; s_links,更新后的s_links
    '''
    for i in range(len(s_nodes)):
        s_nodes[i] += snode_update_matrix[i][0]

    for i in range(len(s_links)):
        u = s_links[i][0]
        v = s_links[i][1]
        s_links[i][2] += slink_update_matrix[u][v]

    return s_nodes, s_links


# 给定一个映射方案mapping solution（包括节点映射方案node mapping solution和链路映射方案link mapping solution），
# 计算他总共的消耗，存在一个cost矩阵中，包括节点资源的cost矩阵，链路资源的cost矩阵。这是一个n×n的邻接矩阵，这样映射这个网络的时候，
# 就减去cost矩阵，释放一个网络的时候就加上他的cost矩阵。
def get_cost_matrix(snode_num, v_nodes, mapping_solution):
    cost_matrix = {
        'snode_cost_matrix': np.zeros(shape=(snode_num, 1), dtype=float),
        'slink_cost_matrix': np.zeros(shape=(snode_num, snode_num), dtype=float)
    }

    node_mapping_solution = mapping_solution['node_mapping_solution']
    link_mapping_solution = mapping_solution['link_mapping_solution']

    # get snode_cost_matrix
    for i in range(len(node_mapping_solution)):
        cost_matrix['snode_cost_matrix'][node_mapping_solution[i]][0] += v_nodes[i]

    # get slink_cost_matrix
    for v_link, s_path in link_mapping_solution.items():
        vlink_bandwidth = v_link[2]
        for i in range(1, len(s_path)):
            u = s_path[i - 1]
            v = s_path[i]
            cost_matrix['slink_cost_matrix'][u][v] += vlink_bandwidth
            cost_matrix['slink_cost_matrix'][v][u] += vlink_bandwidth

    return cost_matrix


# 给定一个虚拟网络的映射方案（包括节点映射和链路映射方案），以比较美观的方式输出这个方案。这是一个输出函数而已
def print_mapping_solution(mapping_solution):
    node_mapping_solution = mapping_solution['node_mapping_solution']
    link_mapping_solution = mapping_solution['link_mapping_solution']

    print('\t\t节点映射方案为')
    print('\t\t            {}'.format(node_mapping_solution))

    print('\t\t链路映射方案为')
    if link_mapping_solution == dict():
        print('\t\t            []')
    else:
        for v_link, s_path in link_mapping_solution.items():
            embedded_s_links = []
            for i in range(1, len(s_path)):
                embedded_s_links.append((s_path[i - 1], s_path[i]))
            print('\t\t            {} -> {}'.format(v_link, embedded_s_links))
