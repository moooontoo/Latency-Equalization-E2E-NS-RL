import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ssl = [5.62, 5.98, 6.1, 6.22, 6.3]
# qoe = [0.76, 0.74, 0.68, 0.60, 0.55]
ssl = [6.93, 7.0, 7.15, 7.2, 7.37]
qoe = [0.71, 0.65, 0.63, 0.625, 0.63]

se = []


x = np.linspace(0, 1, 5)  #alpha
ssl = np.array(ssl)
qoe = np.array(qoe)


#smooth the line
# xlabel = np.linspace(x.min(), x.max(), 300)
xlabel = x
# ssl = make_interp_spline(x, ssl)(xlabel)
# qoe = make_interp_spline(x, qoe)(xlabel)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(xlabel, ssl, color ='blue',linewidth =1.0, linestyle = '--',marker = '>',label = 'SSL')
ax1.legend(loc = 1)

#plt.legend(bbox_to_anchor=(0.83, 0.12), loc=3, borderaxespad=0)
ax1.set_ylabel("Service Satisfaction Level(SSL)")
ax1.set_xlabel("Attention Coefficient Alpha")

#plt.title("SE in Reward")
ax2 = ax1.twinx()
ax2.plot(xlabel, qoe,color ='red',linewidth =1.0, linestyle = '--',marker='^',label = 'QoE')
# plt.yticks(y1)
ax2.legend(loc=4)
plt.ylabel("QoE")
#plt.grid()
plt.title('SSL and QoE Changes Under Different Alpha')
#plt.savefig('SSL and QoE Changes Under Different Alpha.png')
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.savefig('SSL and QoE Changes Under Different Alpha.png', bbox_inches ='tight', dpi = 600, pad_inches = 0.0)
plt.show()