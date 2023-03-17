# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

class Ptr_Net(nn.Module):
    def __init__(self, hidden_size=128, embedding_size=128, num_directions=2,
                 input_size=1, batch_size=128, initialization_stddev=0.1,
                 dropout_p=0, penalty=1e6, s_input_information = {}, use_neighbour_link_penalty =False,device='cpu'):
        super(Ptr_Net, self).__init__()
        if s_input_information['snode resource'] or s_input_information['snode index']:
            input_size = 1
        else:
            input_size = 3
        self.use_neighbour_link_penalty = use_neighbour_link_penalty
        # Define Embedded
        self.Embed = torch.nn.Linear(input_size, embedding_size, bias=False)      #对输入的每个物理节点进行线性变换，并将其嵌入信息传递给encoder
        # Define Encoder
        self.Encoder = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True,
                                     bidirectional=True)
        # Define Attention
        self.W_ref = torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.W_q = torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.v = torch.nn.Linear(num_directions * hidden_size, 1, bias=False)
        # Define Decoder
        self.Decoder = torch.nn.LSTM(input_size=embedding_size * 2, hidden_size=hidden_size, batch_first=True,
                                     bidirectional=True)
        self.DropOut1 = nn.Dropout(p=dropout_p)
        self.DropOut2 = nn.Dropout(p=dropout_p)
        self.W_ref2 = torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.W_q2 = torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.v2 = torch.nn.Linear(num_directions * hidden_size, 1, bias=False)
        self.Softmax_Cross_Entrophy = torch.nn.CrossEntropyLoss(reduction='none')   #reduction：用来指定损失结果返回的是mean、sum还是none
        self.penalty = penalty
        self.s_input_information = s_input_information
        self.device = device

    def get_CrossEntropyLoss(self, output_weights, test_node_mappings):
        test_node_mappings = test_node_mappings.astype(float)  # numpy强制类型转换，04182150
        test_node_mappings = torch.LongTensor(test_node_mappings).to(self.device)   #torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
        v_node_num = test_node_mappings.size()[1]
        path_loss = 0
        for i in range(v_node_num):
            path_loss += self.Softmax_Cross_Entrophy(
                output_weights[i],
                test_node_mappings[:, i].T.squeeze()
            )
        return path_loss

    def get_node_mapping(self, s_node_indexes, s_inputs, v_input):
        batch_size = s_node_indexes.size()[0]
        s_node_num = s_node_indexes.size()[1]
        v_node_num = v_input.size()[0]  # v_node_num
        cannot_penalty = self.penalty

        # Embedding
        # s_node_indexes:(batch,s_node_num,1)
        if self.s_input_information['snode resource']:  # 输入信息仅为s_node_resource
            S_node_Embedding = self.Embed(s_inputs[:, :, 0].unsqueeze(dim=2))

        if self.s_input_information['snode index']:  # 输入信息仅为s_node_index
            S_node_Embedding = self.Embed(s_node_indexes.float())

        # 输入信息为s_node_resource和s_node_neighbour_link_resource
        if self.s_input_information['snode resource and neighbour link resource']:
            S_node_Embedding = self.Embed(s_inputs)

        '''
        Encoder
        S_node_Embedding:(batch,s_node_num,embedding=128)
        '''
        Enc, (hn, cn) = self.Encoder(S_node_Embedding, None)

        '''
        Attention and Decoder
        Enc:(batch, s_node_num, num_directions * hidden_size)
        hn: (batch,num_layers * num_directions,  hidden_size)
        cn: (batch,num_layers * num_directions,  hidden_size)
        '''
        decoder_input = torch.zeros(Enc.size()[0], 1, Enc.size()[2]).to(self.device)
        decoder_state = (hn, cn)
        already_played_actions = torch.zeros(Enc.size()[0], s_node_num).to(self.device)
        decoder_outputs = []
        output_weights = []

        for i in range(v_node_num):

            # Decoder是一个lstm单元, 输入是encoder的输出e0
            decoder_output, decoder_state = self.Decoder(decoder_input, decoder_state)
            decoder_output = self.DropOut2(decoder_output)

            Enc = self.DropOut1(Enc)

            # 判断结点是否满足,对s_node进行变形，对应s_inputs排序，然后再torch.lt
            nodes_without_enough_cpu = torch.lt(s_inputs[:, :, 0], v_input[i][0])  # <
            cannot_satisfy_nodes = nodes_without_enough_cpu
            if self.use_neighbour_link_penalty:
                nodes_without_enough_bandwidth = torch.lt(s_inputs[:, :, 1], v_input[i][1])  #1 放的是临边的和
                nodes_without_enough_bandwidth += torch.lt(s_inputs[:, :, 2], v_input[i][2]) #2 放的是最大的
                cannot_satisfy_nodes += nodes_without_enough_bandwidth

            cannot_node = cannot_satisfy_nodes + already_played_actions


            # 输入 e0 和 decoder的输出
            # output_weight 是decoder的输出，即一个虚拟节点对应的物理节点选择概率向量
            output_weight = torch.squeeze(
                self.v(torch.tanh(
                    self.W_ref(Enc) + self.W_q(decoder_output.repeat(1, s_node_num, 1))
                ))
            ) - cannot_penalty * cannot_node
            output_weights.append(output_weight)

            # 输入 dropout后的e0 和 decoder的输入， 计算出attetion权重，并输出权重
            attention_weight = torch.nn.functional.softmax(
                torch.squeeze(
                    self.v2(torch.tanh(
                        self.W_ref2(Enc) + self.W_q2(decoder_input)
                    ))
                ), dim=1                            #去掉1维的张量
            )

            # 将计算出的attetion权重矩阵 应用于e0, 得到新的e, 新的e作为下一个decoder的输入
            decoder_input = torch.unsqueeze(torch.einsum('ij,ijk->ik', attention_weight, Enc), dim=1)      #batch矩阵相乘

            decoder_outputs.append(torch.argmax(output_weight, dim=1))      #按照行取最大值索引返回
            selected_actions = torch.zeros(Enc.size()[0], s_node_num).to(self.device)
            selected_actions = selected_actions.scatter_(1, torch.unsqueeze(decoder_outputs[-1], dim=1), 1)   #在指定位置写上1
            already_played_actions += selected_actions

        shuffled_node_mapping = np.array([list(item.cpu()) for item in decoder_outputs]).T

        original_node_mapping = np.zeros(shape=(batch_size, v_node_num), dtype=int)
        for i in range(batch_size):
            for j in range(v_node_num):
                original_node_mapping[i][j] = s_node_indexes[i][shuffled_node_mapping[i][j]]

        return original_node_mapping, shuffled_node_mapping, output_weights  # 返回值都是numpy array ##signal

def weights_init(m):
    if isinstance(m, torch.nn.LSTM):      #判断m  是不是 torch.nn.LSTM类型
        torch.nn.init.uniform_(m.weight_ih_l0.data, a=-0.08, b=0.08)    #torch.nn.init.uniform(tensor, a=0, b=1)   从均匀分布U(a, b)中生成值，填充输入的张量或变量
        torch.nn.init.uniform_(m.weight_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_ih_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_hh_l0_reverse.data, a=-0.08, b=0.08)
    else:
        try:
            torch.nn.init.uniform_(m.weight.data, a=-0.08, b=0.08)
            torch.nn.init.uniform_(m.bias.data, a=-0.08, b=0.08)
        except Exception:
            1 + 1
