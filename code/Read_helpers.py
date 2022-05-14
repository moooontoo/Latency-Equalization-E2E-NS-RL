# -*- coding: utf-8 -*-

import numpy as np
import re
seed = 100
np.random.seed(seed)

def get_SN_Node(file): #从文件中读取物理网络的结点，返回以时序为key，以矩阵为value的字典
    SN_dict={}
    fo=open(file,"r")
    pattern=re.compile(r'This is PS network for virtual network----')
    line=fo.readline()
    while line:
        if re.match(pattern,line):#匹配pattern语句
            #line=fo.readline() #读取下一行
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #提取数字存为list   pattern后面的序号
            key=num[0] 
            value=np.zeros([148,148]) #定义一个空矩阵
            for i in range(148):  #读取结点存入矩阵   148个点的节点资源
                line=fo.readline()
                num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]
                value[num[0]][num[0]]=num[1]  #对角矩阵
            SN_dict.update({key:value})#存入字典     key 是pttern后面的序号
        line=fo.readline()
    fo.close()
    return SN_dict

def get_VN_Node(file):#从文件中读取虚拟网络的结点，返回以时序为key，以矩阵为value的字典
    VN_dict={}
    vector=[]
    value=[]
    fo=open(file,"r")
    pattern=re.compile(r'This is  virtual network')#pattern=re.compile(r'This is virtual network----')#开始标志
    end_pattern=re.compile(r'node-link-information:')#结束标志
    line=fo.readline()
    while line:
        if re.match(pattern,line):#匹配pattern语句
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #提取数字存为list
            key=num[0]  
            vector=[]
            value=[]
#            value=np.zeros([10,10]) #定义一个空矩阵
            line=fo.readline()
            while not re.match(end_pattern,line) and line.strip()!='': #读取结点存入矩阵    line.strip() 去除line中头尾包含的空格或换行符
                num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]
                vector.append(num[0])   #虚拟节点编号
                value.append(num[1])    #虚拟节点所需资源数
#                value[num[0]%10][num[0]%10]=num[1] #虚拟网络最大节点数为10
                line=fo.readline()
            size=len(vector)     #VN的虚拟节点数
            values=np.zeros([size,size])
            min_v=min(vector)
            for i in range(len(vector)):      #把虚拟节点所需资源   排布成一个对角阵
                vector[i]=vector[i]-min_v
                values[vector[i]][vector[i]]=value[i]
            VN_dict.update({key:values})
        line=fo.readline()
    fo.close()
    return VN_dict

def get_SN_Link(file,SN_dict):#读取物理网络的边
    fo=open(file,"r")
    SN_key_pattern=re.compile(r'This is PS network for virtual network----')#判断key
    link_pattern=re.compile(r'node-link-information:')#判断link information
    SN_end_pattern=re.compile(r'This is virtual network')#判断结束 结尾处要么是空行 要么是vn
    line=fo.readline()
    while line:
        if re.match(SN_key_pattern,line):#找到SN
            #line=fo.readline() #读取下一行
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #SN的序号
            value=SN_dict.get(key)
            line=fo.readline() #读取下一行
            while not re.match(link_pattern,line):
                line=fo.readline()
            line=fo.readline() #读取下一行
            while (not re.match(SN_end_pattern,line)) and line.strip()!='':
                num=[float(r) for r in re.findall(r"\d+\.?\d*",line)]
                value[int(num[0])][int(num[1])]=num[2]
                value[int(num[1])][int(num[0])]=num[2]
                line=fo.readline() #读取下一行
            SN_dict.update({key:value})
        line=fo.readline()
    fo.close()
    return SN_dict

def get_VN_Link(file,VN_dict):#读取虚拟网路的边
    fo=open(file,"r")
    VN_key_pattern=re.compile(r'This is  virtual network')#判断key
    link_pattern=re.compile(r'node-link-information:')#判断link information
    VN_end_pattern=re.compile(r'The life time is:')#判断结束
    line=fo.readline()
    while line:
        if re.match(VN_key_pattern,line):#找到VN
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #VN的序号
            vector1=[]
            vector2=[]
            value=[]
            
            values=VN_dict.get(key)
            line=fo.readline() #读取下一行
            while not re.match(link_pattern,line):
                line=fo.readline()
            line=fo.readline() #读取下一行
            while not re.match(VN_end_pattern,line) and line.strip()!='':
                #print(line)
                num=[float(r) for r in re.findall(r"\d+\.?\d*",line)]
                vector1.append(num[0])
                vector2.append(num[1])
                value.append(num[2])
                line=fo.readline() #读取下一行
            if len(vector1)>0:
                min_v1=min(vector1)
                min_v2=min(vector2)
                min_v=min(min_v1,min_v2)
            for i in range(len(vector1)):
                vector1[i]=int(vector1[i]-min_v)
                vector2[i]=int(vector2[i]-min_v)
                values[vector1[i]][vector2[i]]=value[i]
                values[vector2[i]][vector1[i]]=value[i]
                
            VN_dict.update({key:values})
        line=fo.readline()
    fo.close()
    return VN_dict

def get_SN_Path(file):#读取物理网络的边，按三元组方式存
    SN_path={}
    fo=open(file,"r")
    SN_key_pattern=re.compile(r'This is PS network for virtual network----')#判断时序
    link_pattern=re.compile(r'node-link-information:')#判断link information
    SN_end_pattern=re.compile(r'This is virtual network')#判断结束 结尾处要么是空行 要么是vn
    line=fo.readline()
    while line:
        if re.match(SN_key_pattern,line):#找到SN
            #line=fo.readline() #读取下一行
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #SN的序号
            value=[]
            line=fo.readline() #读取下一行
            while not re.match(link_pattern,line):
                line=fo.readline()
            line=fo.readline() #读取下一行
            while (not re.match(SN_end_pattern,line)) and (line!='\n'):
                num=[float(r) for r in re.findall(r"\d+\.?\d*",line)]    #读取 两个节点的序号  两节点之间带宽
                node_list=[int(num[0]),int(num[1]),num[2]]
                value.append(node_list)
                line=fo.readline() #读取下一行
            SN_path.update({key:value})
        line=fo.readline()
    fo.close()
    return SN_path

def get_VN_Path(file):#读取虚拟网络的边，按三元组方式存
    VN_path={}
    fo=open(file,"r")
    VN_key_pattern=re.compile(r'This is  virtual network')#判断时序
    link_pattern=re.compile(r'node-link-information:')#判断link information
    VN_end_pattern=re.compile(r'The life time is:')#判断结束
    line=fo.readline()
    while line:
        if re.match(VN_key_pattern,line):#找到VN
            #line=fo.readline() #读取下一行
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #VN的序号
            vector1=[]
            vector2=[]
            value=[]
            values=[]
            
            line=fo.readline() #读取下一行
            while not re.match(link_pattern,line):
                line=fo.readline()
            line=fo.readline() #读取下一行
            while (not re.match(VN_end_pattern,line)) and (line!='\n'):
                num=[float(r) for r in re.findall(r"\d+\.?\d*",line)]
                vector1.append(num[0])          #相连的两节点中1个
                vector2.append(num[1])          #相连的两节点中1个
                value.append(num[2])            #两节点  虚拟带宽
                
#                node_list=(int(num[0]%6),int(num[1]%6),num[2])     #固定6个节点
#                values.append(node_list)
                line=fo.readline() #读取下一行
            if len(vector1)>0:     #存在虚拟链路
                min_v1=min(vector1)
                min_v2=min(vector2)
                min_v=min(min_v1,min_v2)
            for i in range(len(vector1)):
                vector1[i]=int(vector1[i]-min_v)    #和VN节点的编号对应起来
                vector2[i]=int(vector2[i]-min_v)
                node_list=(vector1[i],vector2[i],value[i])
                values.append(node_list)
            VN_path.update({key:values})
        line=fo.readline()
    fo.close()
    return VN_path

def get_Solution(file):#从  maprecord 文件中读取的映射结果  节点映射结果
    MP_dict={}
    fo=open(file,"r")
    pattern=re.compile(r'This is MP Solution for virtual network')#开始标志
    #bad_pattern=re.compile(r'.*\[.*\].*')#不读入的行
    end_pattern=re.compile(r'link MP Solution:')#结束标志
    line=fo.readline()
    while line:
        if re.match(pattern,line):
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #MP的序号
            value=np.zeros([148,10])   #VN 的虚拟节点都在10个以内   节点映射结果
            line=fo.readline() #读取下一行
            while not re.match(end_pattern,line) :#and not re.match(bad_pattern,line):
                num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
                value[num[1]][num[0]%10]=1
                line=fo.readline() #读取下一行
            MP_dict.update({key:value})
        line=fo.readline() #读取下一行
    fo.close()
    return MP_dict

def get_period(file):#从文件中读取虚拟网络的生命周期，返回VN_period={时序key:[[序号num,生命周期period,结束时间end_time],]}
    VN_path={}
    fo=open(file,"r")
    VN_key_pattern=re.compile(r'This is  virtual network')#判断序号
    time_pattern=re.compile(r'The life time is:')#判断life time
    line=fo.readline()
    value=[]
    while line:
        if re.match(VN_key_pattern,line):#找到VN
            #line=fo.readline() #读取下一行
            num=[int(r) for r in re.findall(r"\d+\.?\d*",line)]  #将一行数字处理成list
            key=num[0] #VN的序号
            while not re.match(time_pattern,line):
                line=fo.readline()
            life_time=[int(r) for r in re.findall(r"\d+\.?\d*",line)]
            item=[key,life_time[0],0,0]
            value.append(item)
        line=fo.readline() #读取下一行
    VN_period={0:value}
    fo.close()
    return (VN_period)



def read_SN_VN(file1,file2):    
    solution=get_Solution(file1)
    sn_link=get_SN_Link(file1,get_SN_Node(file1))
    vn_link=get_VN_Link(file2,get_VN_Node(file2))
    sn_path=get_SN_Path(file1)
    
    SN_Link=sn_path[0]
    VN_Link=get_VN_Path(file2)
    VN_Life=get_period(file2)
    SN_Node=[]
    for i in range(len(sn_link[0])):
        SN_Node.append(sn_link[0][i][i])
    VN_Node={}
    for i in range(len(vn_link)):
        v_node=[]
        for j in range(len(vn_link[i])):
            v_node.append(vn_link[i][j][j])
        VN_Node.update({i:v_node})
    return (solution,SN_Link,SN_Node,VN_Link,VN_Node,VN_Life)

import re
def get_CostRatio_UtilizationRate(file):#从文件中读取资源利用率和成本比，返回CostRatio和UtilizationRate两个字典
    fo=open(file,"r",encoding="gbk")
    line=fo.readline()
    #print(type(line))
    pattern1=re.compile(r'.*当前接受的虚拟网络数为.*')#匹配有用行
    pattern2=re.compile(r'(?<=-)\d+(?=\.\d-当前接受的虚拟网络数为)')#匹配第一个数字（时序）
    #pattern5=re.compile(r'(?#-\d+\.\d)(?<=-当前接受的虚拟网络数为-)\d+(?=-)')#匹配网络的序号
    pattern3=re.compile(r'(?#-\d+\.\d-当前接受的虚拟网络数为-\d+-total fitness is:-\d+\.\d+-total cost is-\d+\.\d+-benifit-)(?<=cost ratio is:-)\d+\.\d+(?=-)')#匹配cost_ratio
    pattern4=re.compile(r'(?#-\d+\.\d-当前接受的虚拟网络数为-\d+-total fitness is:-\d+\.\d+-total cost is-\d+\.\d+-benifit-cost ratio is:-\d+\.\d+)(?<=-Utilization rate is:-)\d+\.\d+(?=-)')#匹配utilization_rate
    CostRatio={}
    UtilizationRate={}
    while line:
        #print(line)
        if re.match(pattern1,line):
            #print("ok")
            key=int(re.findall(pattern2,line)[0])#读取时序作为键值
            print(key)
            if key not in CostRatio:
                Cvalue=[]
                CostRatio.update({key:Cvalue})
            if key not in UtilizationRate:
                Uvalue=[]
                UtilizationRate.update({key:Uvalue})
            #num=int(re.findall(pattern4,line)[0])#序号
            cost_ratio=float(re.findall(pattern3,line)[0])
            utilization_rate=float(re.findall(pattern4,line)[0])
            #Cvalue_item=[num,cost_ratio]
            #Uvalue_item=[num,utilization_rate]
            #CostRatio[key].append(Cvalue_item)
            #UtilizationRate[key].append(Uvalue_item)
            CostRatio[key].append(cost_ratio)
            UtilizationRate[key].append(utilization_rate)
        line=fo.readline()
    fo.close()
    return CostRatio, UtilizationRate
