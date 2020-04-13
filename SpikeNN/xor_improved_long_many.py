# for i in range(0,2):
#     for j in range(0,2):
#         print("xor {0} {1} = {2}".format(i, j, bool(i) != bool(j)))
import matplotlib.pyplot as plt
import numpy as np
import math
n = 4000
timeDelta = 0.5  # 5ms
a = 0.02
b = 0.2
d = 8
c = -65

C = 0.0083
T = n
nin = 1

taug = 10.0
frate = 0.002 #[ms^(-1)]

def calcInputCurrent(p1, p2, R):
    return (p1 + p2) / R

class Neuron:
    def __init__(self, in_syn=1, weigth = None):
        self.v = np.zeros((n, 1))
        self.u = np.zeros((n, 1))
        self.v[:, 0] = -70
        self.u[:, 0] = -14
        self.gin = np.zeros((in_syn, 1))
        self.ein = np.zeros((in_syn, 1))
        self.win = 0.07 * np.ones((1, in_syn))
        self.p = np.zeros((n, 1))
        self.p_sim = np.zeros((n, 1))
        self.weigth = weigth
        self.last_fired = -1000
        self.fired = -1000
        self.size = in_syn

    def calc(self, t):
        self.gin = self.gin + self.p[t]
        ginEin = (self.gin * self.ein)
        Iapp = np.matmul(self.win, ginEin)
        Iapp = Iapp - np.matmul(self.win, self.gin) * self.v[t]
        self.gin = (1 - timeDelta / taug) * self.gin

        if self.v[t] < 35:
            dv = (0.04 * self.v[t] + 5) * self.v[t] - self.u[t] + 140
            self.v[t + 1] = self.v[t] + timeDelta * (dv + Iapp)
            self.u[t + 1] = self.u[t] + timeDelta * a * (b * self.v[t] - self.u[t])


        else:
            self.v[t] = 35
            self.v[t + 1] = c
            self.u[t + 1] = self.u[t] + d
            if self.last_fired + 10 > t:
                self.last_fired = self.fired
            else:
                self.last_fired = -1000
            self.fired = t

    def setCurrent(self, cur, t, synaps, cur_zero):

        w = self.weigth[synaps]
        self.p[t] = np.float(abs(self.p[t] - cur)) * w + C * np.float(abs(self.p[0] - cur_zero))

    def lastFired(self):
        return self.last_fired

    def currentFired(self):
        return self.fired

v0 = Neuron(1, np.array([1]))
v1 = Neuron(1, np.array([1]))

weight = 100
v00 = Neuron(4, np.array([1, 0, 1, 0]) / weight)
v01 = Neuron(4, np.array([1, 0, 0, 1]) / weight)
v10 = Neuron(4, np.array([0, 1, 1, 0]) / weight)
v11 = Neuron(4, np.array([0, 1, 0, 1]) / weight)

v_false = Neuron(4, np.array([1, 0, 0, 1]) / weight)
v_true = Neuron(4, np.array([0, 1, 1, 0]) / weight)

v0_sim_fz = 100
v1_sim_fz = v0_sim_fz * 2
for t in range(T-1):
    if t % 1000 == 0:
        v0_sim_fz = v0_sim_fz * 2
        #v1_sim_fz = v1_sim_fz * 2

    if int(math.floor(t * timeDelta % v0_sim_fz)) == 0:
        v0.setCurrent(1, t, 0 ,0)
    elif int(math.floor(t * timeDelta % v0_sim_fz)) == int(math.floor(v0_sim_fz / 2)):
        v0.setCurrent(1, t, 0, 0)
    else:
        v0.setCurrent(0, t, 0, 0)

    if int(math.floor(t * timeDelta % v1_sim_fz)) == 0:
        v1.setCurrent(1, t, 0, 0)
    elif int(math.floor(t * timeDelta % v1_sim_fz)) == int(math.floor(v1_sim_fz / 2)):
        v1.setCurrent(1, t, 0, 0)
    else:
        v1.setCurrent(0, t, 0,0)

    if int(t * timeDelta % v0_sim_fz) < int(v0_sim_fz / 2):
        v0.p_sim[t] = 0
    elif int(t * timeDelta % v0_sim_fz) >= int(v0_sim_fz / 2):
        v0.p_sim[t] = 1

    if int(t * timeDelta % v1_sim_fz) < int(v1_sim_fz / 2):
        v1.p_sim[t] = 0
    elif int(t * timeDelta % v1_sim_fz) >= int(v1_sim_fz / 2):
        v1.p_sim[t] = 1

    v0.calc(t)
    v1.calc(t)

    if v0.p_sim[t] == 0 and v1.p_sim[t] == 0:
        if abs(v0.currentFired() - t) == 0:
            v00.setCurrent(v0.v[t], t, 0, v0.p[0])
        elif abs(v1.currentFired() - t) == 0:
            v00.setCurrent(v1.v[t], t, 2,v1.p[0])

    v00.calc(t)
    if v0.p_sim[t] == 1 and v1.p_sim[t] == 0:
        if abs(v0.currentFired() - t) == 0:
            v01.setCurrent(v0.v[t], t, 0, v0.p[0])
        elif abs(v1.currentFired() - t) == 0:
            v01.setCurrent(v1.v[t], t, 3, v1.p[0])

    v01.calc(t)

    if v0.p_sim[t] == 0 and v1.p_sim[t] == 1:
        if abs(v0.currentFired() - t) == 0:
            v10.setCurrent(v0.v[t], t, 1, v0.p[0])
        elif abs(v1.currentFired() - t) == 0:
            v10.setCurrent(v1.v[t], t, 2, v1.p[0])

    v10.calc(t)
    if v0.p_sim[t] == 1 and v1.p_sim[t] == 1:
        if abs(v0.currentFired() - t) == 0:
            v11.setCurrent(v0.v[t], t, 1, v0.p[0])
        elif abs(v1.currentFired() - t) == 0:
            v11.setCurrent(v1.v[t], t, 3, v1.p[0])

    v11.calc(t)
    if v00.v[t] == 35:
        v_true.setCurrent(v00.v[t], t, 0, v00.p[0])
        v_false.setCurrent(v00.v[t], t, 0, v00.p[0])
    if v01.v[t] == 35:
        v_true.setCurrent(v01.v[t], t, 1, v01.p[0])
        v_false.setCurrent(v01.v[t], t, 1, v01.p[0])
    if v10.v[t] == 35:
        v_true.setCurrent(v10.v[t], t, 2, v10.p[0])
        v_false.setCurrent(v10.v[t], t, 2, v10.p[0])
    if v11.v[t] == 35:
        v_true.setCurrent(v11.v[t], t, 3, v11.p[0])
        v_false.setCurrent(v11.v[t], t, 3, v11.p[0])

    v_true.calc(t)
    v_false.calc(t)

neuron_data = [v0, v1, v00, v01, v10, v11]

fig, axs = plt.subplots(2)
for i in range(0, 2):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:orange',
                np.arange(0, timeDelta * T, timeDelta), neuron_data[i].u, 'tab:green')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

fig, axs = plt.subplots(4)
for i in range(2, 6):
    axs[i-2].plot(np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:orange', np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:green')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()


plt.plot(np.arange(0, timeDelta * T, timeDelta), v00.v + 70, 'tab:green', label = 'v00')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v01.v - 50, 'tab:blue',label = 'v01')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v10.v - 50, 'tab:red',label = 'v10')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v11.v + 70, 'tab:brown',label = 'v11')
plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v0.p_sim + 190, label = "in0")
plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v1.p_sim + 130, label = 'in1')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.legend(loc="upper left")
plt.show()

plt.figure(figsize=(16,8))
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_true.v - 50, 'tab:orange',label = 'v_1')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_true.u -50, 'tab:green',label = 'u_1')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.v + 70, 'tab:cyan',label = 'v_0')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.u + 70, 'tab:blue',label = 'u_0')

plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v0.p_sim + 190, label = 'in0')
plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v1.p_sim + 130, 'tab:red', label = 'in1')
# plt.grid(wchich ='minor', color='grey', linestyle=':', linewidth=1)
plt.xticks(np.arange(0, n/2, step=v0_sim_fz/2))

plt.grid(True)

plt.title('last neuron')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.legend(loc="upper left")
# plt.figsize((16,9))
plt.show()