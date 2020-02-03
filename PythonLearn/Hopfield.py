
import numpy as np
import neurolab as nl

#Zbiór danych wejściowych to litery: N, E, R, O.
#Każda z liter jest zapisywana w macierzy 5x5
target =  [[1,0,0,0,1,
           1,1,0,0,1,
           1,0,1,0,1,
           1,0,0,1,1,
           1,0,0,0,1],
          [1,1,1,1,1,
           1,0,0,0,0,
           1,1,1,1,1,
           1,0,0,0,0,
           1,1,1,1,1],
          [1,1,1,1,0,
           1,0,0,0,1,
           1,1,1,1,0,
           1,0,0,1,0,
           1,0,0,0,1],
          [0,1,1,1,0,
           1,0,0,0,1,
           1,0,0,0,1,
           1,0,0,0,1,
           0,1,1,1,0]]

chars = ['N', 'E', 'R', 'O']
target = np.asfarray(target)
target[target == 0] = -1

#Tworzymy i testujemy sieć Hopfielda
net = nl.net.newhop(target)

#Symulujemy dla elementów zbioru uczącego
output = net.sim(target)
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

#Symulujemy dla zniekształconej litery N
test =np.asfarray([0,0,0,0,0,
                   1,1,0,0,1,
                   1,1,0,0,1,
                   1,0,1,1,1,
                   0,0,0,1,1])
test[test==0] = -1
out = net.sim([test])
print ((out[0] == target[0]).all(), 'Liczba krokow'
       ,len(net.layers[0].outs))