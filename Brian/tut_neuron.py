# https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html
from brian2 import *
import matplotlib.pyplot as plt


start_scope()

tau = 10*ms
eqs = '''
dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
'''

# Change to Euler method because exact integrator doesn't work here
G = NeuronGroup(1, eqs, method='euler')
M = StateMonitor(G, 'v', record=0)

G.v = 1 # initial value

run(100*ms)

plt.plot(M.t/ms, M.v[0])
plt.xlabel('Time (ms)')
plt.ylabel('v');

plt.show()