import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# osor = [14,14,16,16,16,16,17,17,18,18]
# okor = [12,14,14,14,14,15,17,17,18,18]
# static = [10,11,12,11,11,11,13,13,14,16]
# x = np.arange(1, 11,1)

DSDP = [14, 16, 16, 20, 20]
DTDP = [11, 13, 14, 16, 17]
static = [10,11,12,13,15]
DSDP_ = [14, 16, 16, 20, 20]
DTDP_ = [3, 4, 14, 15, 16]
static_ = [3,4,4,8,11]


x = np.array([1,3,5,7,10])
DSDP = np.array(DSDP)
DTDP = np.array(DTDP)
static = np.array(static)


#smooth the line
# xlabel = np.linspace(x.min(), x.max(), 300)
xlabel = x
# ssl = make_interp_spline(x, ssl)(xlabel)
# qoe = make_interp_spline(x, qoe)(xlabel)

fig = plt.figure()
plt.plot(x, DSDP, color ='red', linewidth =1.0, linestyle ='--', marker ='>', label ='DSDP')
plt.plot(x, DTDP, color ='blue', linewidth =1.0, linestyle ='--', marker ='<', label ='DTDP')
plt.plot(x,static, color ='black',linewidth =1.0, linestyle = '--',marker = '*',label = 'static')
plt.legend(loc='best')
plt.ylabel('E2E QoE')
plt.xlabel('URLLC users')
plt.savefig('compare_differ_policy.jpg')
plt.show()