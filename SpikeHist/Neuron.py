import numpy as np
class Neuron:

    def __init__(self, input, n, in_syn=1, weigth = None):
        self.timeDelta = input.timeDelta
        self.taug = input.taug
        self.a = input.a
        self.b = input.b
        self.c = input.c
        self.d = input.d

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
        self.u_step = np.zeros((n, 1))
        self.u_step[:, 0] = -14

    def calc(self, t):
        self.gin = self.gin + self.p[t]
        ginEin = (self.gin * self.ein)
        Iapp = np.matmul(self.win, ginEin)
        Iapp = Iapp - np.matmul(self.win, self.gin) * self.v[t]
        self.gin = (1 - self.timeDelta / self.taug) * self.gin

        if self.v[t] < 35:
            dv = (0.04 * self.v[t] + 5) * self.v[t] - self.u[t] + 140
            self.v[t + 1] = self.v[t] + self.timeDelta * (dv + Iapp)
            self.u[t + 1] = self.u[t] + self.timeDelta * self.a * (self.b * self.v[t] - self.u[t])
            self.u_step[t + 1] = self.u_step[t]

        else:
            self.v[t] = 35
            self.v[t + 1] = self.c
            self.u[t + 1] = self.u[t] + self.d
            self.u_step[t+1] = self.u_step[t] + self.d
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


class InputNeuronData:
    def __init__(self, timeDelta, taug, a, b, c, d):
        self.timeDelta = timeDelta
        self.taug = taug
        self.a = a
        self.b = b
        self.c = c
        self.d = d