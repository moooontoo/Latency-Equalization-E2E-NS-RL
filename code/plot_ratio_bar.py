import matplotlib.pyplot as plt
import numpy as np

# 输入统计数据
slices = ('eMBB1', '2', '3', '4', '5','6', '7', '8', '9', '10',
          'URLLC1','2','3','4','5','6','7','8','9','10')
static = [0.5]*20
osor = [0.6, 0.55, 0.55, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5, 0.65, 0.45, 0.5, 0.5, 0.45, 0.55,0.5,0.5,0.45,0.5,0.5]
same2 = [0.55]*10 + [0.45]*10

bar_width = 0.25  # 条形宽度
index_static = np.arange(len(slices))  # 男生条形图的横坐标
index_osor = index_static + bar_width  # 女生条形图的横坐标
index_same2 = index_osor + bar_width

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
#调整两个子图之间的距离

plt.subplot(211)
# 使用两次 bar 函数画出两组条形图
plt.bar(index_static, height=static, width=bar_width, color='b', label='static')
plt.bar(index_osor, height=osor, width=bar_width, color='r', label='DSDP')
plt.bar(index_same2, height=same2, width=bar_width, color='g', label='DTDP')

plt.legend()  # 显示图例
plt.xticks(index_static + bar_width/2 + 0.15, slices)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.xticks(rotation=70) #倾斜x轴刻度
plt.tick_params(labelsize=7)

plt.ylabel('RAN ratio')  # 纵坐标轴标题
plt.title('The RAN delay ratio of E2E delay')  # 图形标题
plt.ylim((0.35,0.75))


plt.subplot(212)
# 使用两次 bar 函数画出两组条形图
plt.bar(index_static, height=static, width=bar_width, color='b', label='static')
plt.bar(index_osor, height=osor, width=bar_width, color='r', label='DSDP')
plt.bar(index_same2, height=same2, width=bar_width, color='g', label='DTDP')

plt.legend()  # 显示图例
plt.xticks(index_static + bar_width/2 + 0.15, slices)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.xticks(rotation=70) #倾斜x轴刻度
plt.tick_params(labelsize=7)

plt.ylabel('RAN ratio')  # 纵坐标轴标题
plt.title('URLLC')  # 图形标题
plt.ylim((0.35,0.75))


plt.show()
