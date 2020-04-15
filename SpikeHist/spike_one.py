import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from Neuron import *
from InitData import *

neurons_amount = 8
treshold = 50
def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    hist = cv2.calcHist(img, [0], None, [neurons_amount], [0, 256])

    return hist

input_neuron_data = InputNeuronData(timeDelta, taug, a, b, c, d)

# name = 'model'
# path = Path(".")
# path = path.glob('../db/img_spike/'+name+'/*.jpg')

model_imagepath = '../db/img_spike/model/IMG_20200321_182525.jpg'
model_hist = returnHistogram(model_imagepath)



test_imagepath = '../db/img_spike/test/IMG_20200321_182525.jpg'
test_hist = returnHistogram(test_imagepath)



neurons = []
for i in range(neurons_amount):
    neurons.append(Neuron(input_neuron_data, n, 1, np.array([1])))

true_out = Neuron(input_neuron_data, n, neurons_amount, np.ones(neurons_amount))
out = Neuron(input_neuron_data, n, 1, np.array([1]))

max_probe = 200
for t in range(T - 1):
    for i in range(neurons_amount):
        if test_hist[i] == t:
            neurons[i].setCurrent(2, t, 0)
            print("hist "+ str(t) + " "+str(i))
        neurons[i].calc(t)
    if t > 10:
        for j in range(neurons_amount):
            if neurons[j].fired == t and model_hist[j] == t - 11:
                true_out.setCurrent(1, t, j)
                print("fired " + str(t) + " " + str(i))
    true_out.calc(t)

    if t > 500 and t < 505 and true_out.u_step[t] > treshold:
        out.setCurrent(1, t, 0)
    else:
        out.setCurrent(0, t, 0)
    out.calc(t)





plt.figure(figsize=(16,10))
fig, axs = plt.subplots(neurons_amount)
plt.ylabel('voltage [mV]')
#plt.title('histogram spikes')
for i in range(neurons_amount):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].v, label = 'neuron v'+ str(i))
    axs[i].legend(loc="upper right")
    #axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].u, label = 'neuron u'+ str(i))
plt.xlabel('time [ms]')


#plt.figsize((16,9))
plt.show()

plt.plot(model_hist, 'tab:green', label = 'model hist')
plt.plot(test_hist + 50, 'tab:blue', label = 'test hist + offset')

plt.legend(loc="upper left")
plt.show()
plt.figure(figsize=(12,8))
plt.plot(np.arange(0, timeDelta * T, timeDelta), true_out.v, label = 'v hist')
plt.plot(np.arange(0, timeDelta * T, timeDelta), true_out.u, label = 'u step hist')
plt.plot(np.arange(0, timeDelta * T, timeDelta), true_out.u_step, label = 'u step')
plt.plot(np.arange(0, timeDelta * T, timeDelta), out.v, label = 'v_last')
plt.plot(np.arange(0, timeDelta * T, timeDelta), out.u, label = 'u_last')
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.title('counter neuron')
plt.legend(loc="upper left", prop={"size":20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

plt.plot(np.arange(0, timeDelta * T, timeDelta), out.v, label = 'v hist')
plt.plot(np.arange(0, timeDelta * T, timeDelta), out.u, label = 'u')

plt.grid(color='grey', linestyle=':', linewidth=1)
plt.title('last neuron')
plt.legend(loc="upper left")
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

