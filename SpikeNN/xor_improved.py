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
            self.last_fired = self.fired
            self.fired = t

    def setCurrent(self, cur, t):
        self.p[t] = np.average(cur / self.weigth / self.size)

    def lastFired(self):
        return self.last_fired

    def currentFired(self):
        return self.fired

v0 = Neuron(weigth=1)
v1 = Neuron(weigth=1)

v00 = Neuron(4, np.array([1, 4, 1, 4]))
v01 = Neuron(4, np.array([1, 4, 4, 1]))
v10 = Neuron(4, np.array([4, 1, 1, 4]))
v11 = Neuron(4, np.array([4, 1, 4, 1]))

v_true = Neuron(4, np.array([2, 0.25, 0.25, 2]))
v_false = Neuron(4, np.array([0.25, 2, 2, 0.25]))


for t in range(T - 1):
    if t > 30:
        if int(t * timeDelta % 300) < 10:
            v0.setCurrent(1, t)
        elif int(t * timeDelta % 50) < 3:
            v0.setCurrent(1, t)
        else:
            v0.setCurrent(0, t)

        if int(t * timeDelta % 150) < 10:
            v1.setCurrent(1, t)
        elif int(t * timeDelta % 100) < 3:
            v1.setCurrent(1, t)
        else:
            v1.setCurrent(0, t)

    v0.calc(t)
    v1.calc(t)

    p = np.array([0, 0, 0, 0])
    if v0.currentFired() > t - 3:
        if v0.lastFired() > t-10:
            p[0] = 1
        else:
            p[1] = 1
    if v1.currentFired() > t - 3:
        if v1.lastFired() > t-10:
            p[2] = 1
        else:
            p[3] = 1


    v00.setCurrent(p, t)
    v01.setCurrent(p, t)
    v10.setCurrent(p, t)
    v11.setCurrent(p, t)

    v00.calc(t)
    v01.calc(t)
    v10.calc(t)
    v11.calc(t)

    p = np.array([0, 0, 0, 0])
    if v00.v[t] == 35:
        p[0] = 1
    if v01.v[t] == 35:
        p[1] = 1
    if v10.v[t] == 35:
        p[2] = 1
    if v11.v[t] == 35:
        p[3] = 1

    v_true.setCurrent(p, t)
    v_false.setCurrent(p, t)

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
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.v + 20, 'tab:cyan')
plt.plot(np.arange(0, timeDelta * T, timeDelta), v_false.u + 20, 'tab:blue')

plt.plot(np.arange(0, timeDelta * T, timeDelta), 30 * v0.p + 40)
plt.plot(np.arange(0, timeDelta * T, timeDelta), 30 * v1.p - 110)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.title('last neuron')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()