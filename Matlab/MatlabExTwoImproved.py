import matplotlib.pyplot as plt
import numpy as np

n = 2000
timeDelta = 0.5  # 5ms
a = 0.02
b = 0.2
d = 8
c = -65
v = np.zeros((n, 1))
u = np.zeros((n, 1))
v[:, 0] = -70
u[:, 0] = -14

nin = 100
rate = 0.002
taug = 10

gin = np.zeros((nin, 1))
ein = np.zeros((nin, 1))
win = 0.07 * np.ones((1, nin))
p = np.zeros((nin, 1))

Iapp = 7  # pA
Iappp = 0
T = 2000
for t in range(T - 1):
    if t * timeDelta > 200 and t * timeDelta < 700:
        p = np.random.ranf((nin, 1)) < rate * timeDelta

    else:
        p =0

    gin = gin + p
    ginEin = (gin * ein)
    Iapp = np.matmul(win, ginEin)
    Iapp = Iapp - np.matmul(win,gin) * v[t]
    gin = (1 - timeDelta / taug) * gin


    if v[t] < 35:
        dv = (0.04 * v[t] + 5) * v[t] - u[t] + 140
        v[t+1] = v[t] + timeDelta * (dv + Iapp)
        u[t+1] = u[t] + timeDelta * a * (b * v[t] - u[t])

    else:
        v[t] = 35
        v[t+1] = c
        u[t+1] = u[t] + d




# print(v)
plt.plot(np.arange(0, timeDelta * T, timeDelta), v, np.arange(0, timeDelta * T, timeDelta), u)

plt.title('Simulate a single neuron with injected current')
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
plt.show()

# fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# axs[0].plot(np.arange(0, timeDelta * T, timeDelta), v, np.arange(0, timeDelta * T, timeDelta), u, 'tab:green')
# axs[0].set_title("first neuron")
# axs[1].plot(np.arange(0, timeDelta * T, timeDelta), vv, np.arange(0, timeDelta * T, timeDelta), uu, 'tab:red')
# axs[1].set_title("second neuron, activated by first")
# plt.xlabel('time [ms]')
# plt.ylabel('voltage [mV]')
# plt.show()
# plt.plot(np.arange(0, timeDelta * T, timeDelta), vv, np.arange(0, timeDelta * T, timeDelta), uu)
# plt.title('Simulate a single neuron with injected current')
# plt.xlabel('time [ms]')
# plt.ylabel('voltage [mV]')
# plt.show()
