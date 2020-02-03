import numpy as np
import neurolab as nl
import numpy.random as rand
import pylab as pl

#Tworzymy zbiór danych
#Środki klas
centr = np.array([[0.2, 0.2], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5]])
rand_norm = 0.05 * rand.randn(100, 4, 2)
#dane uczące
inp = np.array([centr + r for r in rand_norm])
inp.shape = (100 * 4, 2)
rand.shuffle(inp)

#Tworzymy sieć Kohonena z dwoma wejściami i 4 neuronami
net = nl.net.newc([[0.0, 1.0],[0.0, 1.0]], 4)
#Uczymy sieć zgodnie z algortmem Winner Take All
error = net.train(inp, epochs=200, show=20)

#Wyniki umieszczamy na wykresie:
w = net.layers[0].np['w']
pl.plot(inp[:,0], inp[:,1], '.', \
        centr[:,0], centr[:, 1] , 'yv', \
        w[:,0], w[:,1], 'p')
pl.legend(['obiekty uczace', 'rzeczywiste srodki klas',
           'nauczone srodki klas'])
pl.show()