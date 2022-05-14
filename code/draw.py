# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:56:25 2019

@author: zheng
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:38:09 2018

@author: tianz
"""

def draw():
    import copy
    import matplotlib.pyplot as plt
    import numpy as np

    R_c=[]
    resource=[]
    original_resource=[]
    original_r_c=[]
    for key in R_C:

        a=0
        max_value=0

        result=copy.deepcopy(R_C[key])
        for i in range(1,len(result)):
            if result[-i]>0:
                max_value=result[-i]
                break

        if result == []:
            result.append(R_c[-1])
            max_value=result[-1]
    #    for i in result:
    #        if i>0:
    #            a=a+1
    #            max_value=max_value+i
    #    max_value=(max_value/a)
        R_c.append(max_value)

    for key in CostRatio:
        a=0
        max_value=0

        result=copy.deepcopy(CostRatio[key])
        max_value=result[-1]
    #    for i in result:
    #        if i>0:
    #            a=a+1
    #            max_value=max_value+i
    #    max_value=(max_value/a)
        original_r_c.append(max_value)


    for key in RES:

        result=copy.deepcopy(RES[key])

        #print("result",result)
        #print("R_c",R_c)

        if result == []:

            max_value=resource[-1]

        else:
            for i in range(1,len(result)):
                if result[-i]>0:
                    max_value=result[-i]
                    break
    #    max_value=max(result)
        print(max_value)
        resource.append(max_value)


    for key in UtilizationRate:
        result=copy.deepcopy(UtilizationRate[key])
       # max_value=max(result)
        max_value=result[-1]
        original_resource.append(max_value)

    original_resource=original_resource[0:19]
    original_r_c=original_r_c[0:19]
    x=np.arange(0,950,50)
    #resource=resource[0:32]
    #R_c=R_c[0:32]

    print(len(x),len(resource))
    print(len(x),len(original_resource))

    plt.rcParams['font.sans-serif']=['Arial']  #如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus']=False  #显示负号


    #label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    #color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    #线型：-  --   -.  :    ,
    #marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10,5))
    plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    plt.plot(x,original_resource,color="black",label="VNE-PTIC",linewidth=1.5)
    plt.plot(x,resource,"k--",label="Active Search",linewidth=1.5)
    plt.xlabel("Time",fontsize=15,fontweight='bold')
    plt.ylabel("physical Node Utilization",fontsize=15,fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15,fontweight='bold') #设置图例字体的大小和粗细

    plt.savefig('resource5.svg',format='svg')  #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()



    plt.rcParams['font.sans-serif']=['Arial']  #如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus']=False  #显示负号


    #label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    #color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    #线型：-  --   -.  :    ,
    #marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10,5))
    plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    plt.plot(x,original_r_c,color="black",label="VNE-PTIC",linewidth=1.5)
    plt.plot(x,R_c,"k--",label="Active Search",linewidth=1.5)
    plt.xlabel("Time",fontsize=15,fontweight='bold')
    plt.ylabel("R/C Ratio of Substrate Network",fontsize=15,fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15,fontweight='bold') #设置图例字体的大小和粗细

    plt.savefig('R_c-5.svg',format='svg')  #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()


    import json
    jsObj = json.dumps(R_C)    #把python对象转换成json对象的一个过程，生成的是字符串
    fileObject = open('all_r_c5.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()

    jsObj= json.dumps(RES)
    fileObject = open('all_resource5.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()


def draw_loss():
    # -*- coding: utf-8 -*-
    """
    Created on Mon May 20 22:08:49 2019

    @author: zheng
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import spline
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  #去掉上边框
    # ax.spines['right'].set_visible(False) #去掉右边框
    x = np.arange(0, 200)
    # xnew = np.linspace(x.min(),x.max(),300)
    # power_smooth = spline(x,l,xnew)

    # plt.plot(xnew,power_smooth,color="black",linewidth=1.5)
    plt.plot(x, meanLoss, color="black", linewidth=1.5)
    plt.xlabel("Iteration Steps", fontsize=15, fontweight='bold')
    plt.ylabel("Loss", fontsize=15, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.show()
    plt.savefig('meanloss50_20190603(2).svg', format='svg')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中

    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1

    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(x, meanhop, color="black")
    plt.xlabel("Iteration Steps", fontsize=15, fontweight='bold')
    plt.ylabel("Mean Hops", fontsize=15, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')
    plt.savefig('meanhop50_20190603(2).svg', format='svg')

    x = np.arange(0, 50)
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(x, NodeUtilization_50, linewidth=1.5)
    plt.xlabel("Iteration Steps", fontsize=15, fontweight='bold')
    plt.ylabel("physical Node Utilization", fontsize=15, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.savefig('NodeUtilization50_20190603(2).svg', format='svg')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    # plt.plot(x,original_resource,color="black",label="VNE-PTIC",linewidth=1.5)
    plt.plot(x, RC_50, linewidth=1.5)
    plt.xlabel("Iteration Steps", fontsize=15, fontweight='bold')
    plt.ylabel("R/C Ratio of Substrate Network", fontsize=15, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('RC50_20190603(2).svg', format='svg')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()