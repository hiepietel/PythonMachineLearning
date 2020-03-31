# for i in range(0,2):
#     for j in range(0,2):
#         print("xor {0} {1} = {2}".format(i, j, bool(i) != bool(j)))
import matplotlib.pyplot as plt
import numpy as np

n = 2000
timeDelta = 0.5  # 5ms
a = 0.02
b = 0.2
d = 8
c = -65

T = 2000
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

    def setCurrent(self, cur, t, synaps):

        w = self.weigth[synaps]
        self.p[t] = np.float(abs(self.p[t] - cur)) * w

    def lastFired(self):
        return self.last_fired

    def currentFired(self):
        return self.fired

v0 = Neuron(1, np.array([1]))
v1 = Neuron(1, np.array([1]))

v00 = Neuron(4, np.array([1, 0, 1, 0])/100)
v01 = Neuron(4, np.array([1, 0, 0, 1])/100)
v10 = Neuron(4, np.array([0, 1, 1, 0])/100)
v11 = Neuron(4, np.array([0, 1, 0, 1])/100)

v_false = Neuron(4, np.array([1, 0, 0, 1])/100)
v_true = Neuron(4, np.array([0, 1, 1, 0])/100)


for t in range(T-1):
    if t > 10:
        if int(t * timeDelta % 395) < 6:
            v0.setCurrent(1, t ,0)
        elif int(t * timeDelta % 112) <2:
            v0.setCurrent(1, t, 0)
        else:
            v0.setCurrent(0, t, 0)

        if int(t * timeDelta % 321) < 6:
            v1.setCurrent(1, t, 0)
        elif int(t * timeDelta % 201) <2:
            v1.setCurrent(1, t, 0)
        else:
            v1.setCurrent(0, t, 0)

    v0.calc(t)
    v1.calc(t)

    if v0.currentFired() ==t and v0.lastFired() > t-8:
        v00.setCurrent(v0.v[t],t, 0)
        v01.setCurrent(v0.v[t],t, 0)
        v10.setCurrent(v0.v[t],t, 0)
        v11.setCurrent(v0.v[t],t, 0)
    elif v0.currentFired() ==t and v0.lastFired() == -1000:
        v00.setCurrent(v0.v[t],t, 1)
        v01.setCurrent(v0.v[t],t, 1)
        v10.setCurrent(v0.v[t],t, 1)
        v11.setCurrent(v0.v[t],t, 1)
    elif v1.currentFired() == t and v1.lastFired() > t-8:
        v00.setCurrent(v1.v[t],t, 2)
        v01.setCurrent(v1.v[t],t, 2)
        v10.setCurrent(v1.v[t],t, 2)
        v11.setCurrent(v1.v[t],t, 2)
    elif v1.currentFired() == t and v1.lastFired() == -1000:
        v00.setCurrent(v1.v[t], t, 3)
        v01.setCurrent(v1.v[t], t, 3)
        v10.setCurrent(v1.v[t], t, 3)
        v11.setCurrent(v1.v[t], t, 3)


    v00.calc(t)
    v01.calc(t)
    v10.calc(t)
    v11.calc(t)

    if v00.v[t] == 35:
        v_true.setCurrent(v00.v[t], t, 0)
        v_false.setCurrent(v00.v[t], t, 0)
    if v01.v[t] == 35:
        v_true.setCurrent(v01.v[t], t, 1)
        v_false.setCurrent(v01.v[t], t, 1)
    if v10.v[t] == 35:
        v_true.setCurrent(v10.v[t], t, 2)
        v_false.setCurrent(v10.v[t], t, 2)
    if v11.v[t] == 35:
        v_true.setCurrent(v11.v[t], t, 3)
        v_false.setCurrent(v11.v[t], t, 3)


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
    axs[i - 2].plot(np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:orange',
                    np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:green')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

plt.plot(np.arange(0, timeDelta * T, timeDelta), v_true.v, 'tab:orange')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_true.u, 'tab:green')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.v + 80, 'tab:cyan')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.u + 80, 'tab:blue')

plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v0.p + 130)
plt.plot(np.arange(0, timeDelta * T, timeDelta), 50 * v1.p - 130)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.title('last neuron')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()