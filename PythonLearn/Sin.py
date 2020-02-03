import neurolab as nl
import numpy as np
import pylab as pl

#Tworzymy zbiór ucząc
x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

size = len(x)
print(x)
inp = x.reshape(size,1)
print(inp)
tar = y.reshape(size,1)

#Tworzymy sieć z dwoma warstwami, inicjalizowaną w sposób losowy,
#w pierwszej warstwie 5 neuronów, w drugiej warstwie 1 neuron
net = nl.net.newff([[-7, 7]],[5, 1])

#Uczymy sieć, wykorzystujemy metodę największego spadku gradientu
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)

#Symulujemy
out = net.sim(inp)

#Tworzymy wykres z wynikami
x2 = np.linspace(-6.0,6.0,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = out.reshape(size)

pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()