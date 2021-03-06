import matplotlib.pyplot as plt
import numpy as np

timeDelta = 0.5 #5ms
a = 0.02
b = 0.2
d = 8
c = -65
v = [-70]
u = [-14]
Iapp =7 #pA
T=2000
for t in range(T-1):
    if t*timeDelta > 200 and t*timeDelta < 700:
        Iapp = 7
    else:
        Iapp = 0
    if v[t] < 35:
        dv = (0.04*v[t]+5)*v[t] - u[t] + 140
        v.append(v[t] + timeDelta * (dv + Iapp))
        u.append(u[t] + timeDelta * a * ( b * v[t] - u[t]))
    else:
        v[t] = 35
        v.append(c)
        u.append(u[t]+d)
    
    #plt.plot(t+1, u[t+1])

#print(v)
plt.plot(np.arange(0,timeDelta*T,timeDelta),v, np.arange(0,timeDelta*T,timeDelta), u)
plt.title('Simulate a single neuron with injected current')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()
