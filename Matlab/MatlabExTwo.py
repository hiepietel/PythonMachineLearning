import matplotlib.pyplot as plt
import numpy as np
import random as rnd

timeDelta = 0.5 #5ms
#task1
a = 0.02
b = 0.2
d = 8
c = -65
v = [-70]
u = [-14]
Iapp =7 #pA
T=2000
#task2
taug = 10.0
frate = 0.002 #[ms^(-1)]
nin = 100
#init vectors
gin = [0.0] * nin
Ein = [0.0] * nin
win = [0.07] * nin
for t in range (T-1):
 #   if r <= frate*timeDelta:
 #       p.append(1)
 #   else:
 #       p.append(0)
    p = []
    if 200 <= t * timeDelta <=700:
        for i in range(100):
            p.append(rnd.randint(0,nin)%frate*timeDelta)
    else: 
        p = [0.0] * nin
    for i in range(nin):
        gin[i] = gin[i] + p[i]
    ginEin = 0
    wingin = 0
    for i in range(nin):
        ginEin = ginEin + gin[i] * Ein[i]
        wingin = wingin + (win[i] * gin[i])*v[t]
    Iapp = ginEin - wingin
    for i in range(100):
        gin[i] = (1.0 - timeDelta / taug) * gin[i]
    if v[t] < 35:
        dv = (0.04*v[t]+5)*v[t] - u[t] + 140 + Iapp
        v.append(v[t]+(dv+Iapp)*timeDelta)
        du = a * ( b * v[t] - u[t])
        u.append(u[t] + (timeDelta * du))
    else:
        v[t] = 35
        v.append(c)
        u.append(u[t]+d)

    #plt.plot(t+1, u[t+1])

#print(v)
plt.plot(np.arange(0,timeDelta*T,timeDelta),v, np.arange(0,timeDelta*T,timeDelta), u)
plt.title('Single neuron with synaptic input')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()
