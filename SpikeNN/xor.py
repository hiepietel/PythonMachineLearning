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

class Neuron:
    def __init__(self):
        self.v = np.zeros((n, 1))
        self.u = np.zeros((n, 1))
        self.v[:, 0] = -70
        self.u[:, 0] = -14
        self.gin = np.zeros((nin, 1))
        self.ein = np.zeros((nin, 1))
        self.win = 0.07 * np.ones((1, nin))

    def calc(self, p):
        self.gin = self.gin + p
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

v0 = Neuron()
v1 = Neuron()

v00 = Neuron()
v01 = Neuron()
v10 = Neuron()
v11 = Neuron()

v = Neuron()

p0 = 0
p1 = 0

out_0 = 0
out_1 = 0

for t in range(T - 1):

    if (int(t*timeDelta % 75) < 4):
        p0 = 1
    elif (int(t*timeDelta % 150) < 10):
        p0 = 1
    else:
        p0 = 0


    if(int(t * timeDelta % 50)< 4):
        p1 = 1
    elif (int(t*timeDelta % 100) < 10):
        p1 = 1
    else:
        p1 = 0
    # if t * timeDelta > 0 and t * timeDelta < 250:
    #     p0 = 0
    #     p1 = 0
    #
    # elif t * timeDelta > 250 and t * timeDelta < 500:
    #     p = 0
    #     p[1] = 0.1
    #
    # elif t * timeDelta > 500 and t * timeDelta < 750:
    #     p[0] = 0.1
    #     p[1] = 0
    # else:
    #     p[0] = 0.1
    #     p[1] = 0.1

    v0.calc(p0)
    v1.calc(p1)
    p00 = 0
    p01 = 0
    p10 = 0
    p11 = 0
    border = 2
    amount = 50
    if t > amount and t % 10 == 0:
        first_amount = 0
        for z in range(amount):
            if v0.v[t-z] == 35:
                first_amount += 1
        secont_amount = 0
        for z in range(amount):
            if v1.v[t-z] == 35:
                secont_amount += 1

        if first_amount > 0 and first_amount < border and secont_amount> 0 and secont_amount < border:
            p00 = 1

        if first_amount > 0 and first_amount < 20 and secont_amount>border and secont_amount < 20:

            p01 = 1

        if first_amount > border and secont_amount>0 and secont_amount <border:

            p10 = 1

        if first_amount > border and secont_amount>border:

            p11 = 1


    v00.calc(p00)
    v01.calc(p01)
    v10.calc(p10)
    v11.calc(p11)

    p_end = 0

    if v00.v[t] == 35 or v11.v[t] == 35:
        out_1 = 5 + t
    elif v01.v[t] == 35 or v01.v[t] == 35:
        out_0 = 3 + t

    if out_0 > t:
        p_end = 1
        out_0 += 1
    elif out_1 > t:
        p_end = 1
        out_1 += 1


    v.calc(p_end)
    p_end = 0
neuron_data = [v0, v1, v00, v01, v10, v11]

fig, axs = plt.subplots(2)
for i in range(0, 2):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v,'tab:orange', np.arange(0, timeDelta * T, timeDelta), neuron_data[i].u, 'tab:green')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

fig, axs = plt.subplots(4)
for i in range(2, 6):
    axs[i-2].plot(np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:orange', np.arange(0, timeDelta * T, timeDelta), neuron_data[i].v, 'tab:green')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

plt.plot(np.arange(0, timeDelta * T, timeDelta), v.v, np.arange(0, timeDelta * T, timeDelta), v.u)

plt.title('end simulation')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

# plt.xlabel('time [ms]')
# plt.ylabel('voltage [mV]')
# plt.show()
# # print(v)
# plt.plot(np.arange(0, timeDelta * T, timeDelta), v00, np.arange(0, timeDelta * T, timeDelta), u00)

