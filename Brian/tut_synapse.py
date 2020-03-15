from brian2 import *

import matplotlib.pyplot as plt

start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
G.I = [1, 1]
G.tau = [10, 10]*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=0, j=1)
S.w = 'j*0.2+i*0.2'

M = StateMonitor(G, 'v', record=True)

run(50*ms)

plt.plot(M.t/ms, M.v[0], label='Neuron 0')
plt.plot(M.t/ms, M.v[1], label='Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend()
plt.show()

print("exit")
