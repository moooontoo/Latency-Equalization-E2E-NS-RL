import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

ssl = [6.93, 7.0, 7.15, 7.2, 7.3]
qoe = [0.71, 0.65, 0.63, 0.61, 0.59]

x = np.linspace(0, 1, 5)  #alpha
ssl = np.array(ssl)
qoe = np.array(qoe)


#smooth the line
xlabel = np.linspace(x.min(), x.max(), 300)

ssl = make_interp_spline(x, ssl)(xlabel)
qoe = make_interp_spline(x, qoe)(xlabel)

fig, ax1 = plt.subplots()
plt.grid(True)
ax2 = ax1.twinx()

plt.xlim((0, 1))
# plt.ylim((0, 1))
plt.xlabel('Alpha')

ax1.set_ylabel('Service Satisfaction Level(SSL)')
ax1.set_ylim((6.8, 7.35))


ax2.set_ylabel('QoE')
ax2.set_ylim((0.55, 0.75))
# ax1.plot(xlabel, ssl, color='blue',linewidth =1.0, linestyle = '-.',label = 'SSL')
l1, = plt.plot(xlabel, ssl, color ='blue',linewidth =1.0, linestyle = '--',label = 'SSL')
plt.legend(bbox_to_anchor=(0.83, 0.12), loc=(0.83, 0.589), borderaxespad=0)
ax2.plot(xlabel, qoe, color ='red',linewidth =1.0, linestyle = '--',label = 'QoE')
#plt.legend(loc=(0.83, 0.189))
plt.legend(loc='lower right')

plt.title('SSL and QoE Changes Under Different Alpha')
plt.savefig('./alpha_ssl_qoe.jpg')
plt.show()

