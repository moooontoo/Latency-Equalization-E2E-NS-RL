#matplotlib inline
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt

fig = plt.figure(1)

host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])  #主轴
par1 = ParasiteAxes(host, sharex=host)
par2 = ParasiteAxes(host, sharex=host)
host.parasites.append(par1)
host.parasites.append(par2)

host.set_ylabel('Spectrum Efficiency')
host.set_xlabel('Attention Coefficient Alpha')

host.axis['right'].set_visible(False)
par1.axis['right'].set_visible(True)
par1.set_ylabel('Number of eMBB Users Accessed')

par1.axis['right'].major_ticklabels.set_visible(True)
par1.axis['right'].label.set_visible(True)

par2.set_ylabel('Number of URLLC Users Accessed')
offset = (40, 0)
new_axisline = par2._grid_helper.new_fixed_axis
 # "_grid_helper"与"get_grid_helper()"等价，可以代替
#new_axisline = par2.get_grid_helper().new_fixed_axis  # 用"get_grid_helper()"代替，结果一样，区别目前不清楚
par2.axis['right2'] = new_axisline(loc='right', axes=par2, offset=offset)

fig.add_axes(host)

# host.set_xlim(0,2)
# host.set_ylim(0,2)

host.set_xlabel('Attention Coefficient Alpha')
host.set_ylabel('Spectrum Efficiency(bps/Hz)')
# host.set_ylabel('eMBB users accessed')
se = [5.97, 5.76, 5.86, 5.79, 5.83]
se1 = [5.85, 5.76, 5.83, 5.86, 5.97]
urllc_10 = [9.47,9.04,8.77,8.54,8.34]
urllc_15 = [i *1.5 for i in urllc_10]
p1, = host.plot([0,0.25,0.5,0.75,1], se1, linewidth =1.0, linestyle = '--',marker = '>',label="Spectrum efficiency")
p2, = par1.plot([0,0.25,0.5,0.75,1], [3.98, 4.13, 4.25, 4.5,5],linewidth =1.0, linestyle = '--',marker='^',label="eMBB users accessed")
p3, = par2.plot([0,0.25,0.5,0.75,1], urllc_15,linewidth =1.0, linestyle = '--',marker='^', label="URLLC users accessed")

#par1.set_ylim(8.3,9.6)
#par2.set_ylim(3.90,5)

host.legend()
#轴名称，刻度值的颜色
host.axis['left'].label.set_color(p1.get_color())
par1.axis['right'].label.set_color(p2.get_color())
par2.axis['right2'].label.set_color(p3.get_color())
par2.axis['right2'].major_ticklabels.set_color(p3.get_color()) #刻度值颜色
par2.axis['right2'].set_axisline_style('-|>',size=1.5)  #轴的形状色
par2.axis['right2'].line.set_color(p3.get_color())  #轴的颜色
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
#plt.grid()

plt.savefig('three_zhou.png', bbox_inches ='tight', dpi = 600, pad_inches = 0.0)
plt.show()
#alpha  三个坐标轴的实验对比图