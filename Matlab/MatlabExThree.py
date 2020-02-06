import matplotlib.pyplot as plt
import numpy as np
import random as rnd

timeDelta = 0.5 #5ms
n = 1000
kinh = np.random.ranf(n)
kinh = kinh % 0.2
kexc = np.random.ranf(n)
d = 8 * kexc + 2 * kinh
a = 0.02 * kexc + 0.1 * kinh

T = np.ceil(n/timeDelta)
v = np.zeros(n, T)
u = np.zeros(n, T)
v[0] = -70
u[0] = -14

Iapp =7 #pA
nin = 100
frate = 0.002 #[ms^(-1)]
taug = 10.0
gin = np.zeros(nin)
Ein = np.zeros(nin)
Win = np.full((n, nin), 0.07)

for t in range(n-1):
    if t * timeDelta > 200 and t * timeDelta < 700:
        p = np.random.ranf(nin)
        p = p % frate * timeDelta
    else:
        p = 0
    gin = gin + p
    Iapp = Win * (gin*Ein)
    #Iapp = Iapp - (Win * gin) * v
    gin = (1 - timeDelta/taug) * gin


plt.plot(np.arange(0,timeDelta*n,timeDelta),v, np.arange(0,timeDelta*n,timeDelta), u)
plt.title('Single neuron with synaptic input')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()
