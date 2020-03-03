# https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html
from brian2 import *

start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs)
run(100*ms)