import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from Neuron import *
from InitData import *

neurons_amount = 4
treshold = (-14 + neurons_amount *d) * 0.5
T=1000
def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    hist = cv2.calcHist(img, [0], None, [neurons_amount], [0, 256])

    return hist

def returnSky(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    img = cv2.resize(img, (neurons_amount, height))
    # plt.imshow(img)
    # plt.show()
    image_part = np.split(img, int(3))[0]
    b, g, r, = cv2.split(image_part)
    i = 0
    sky_hist = np.empty((width, 1))
    for col in b.T:
        average = np.average(col)
        sky_hist[i] = math.floor(average)
        i += 1
    return sky_hist

# name = 'model'
# path = Path(".")
# path = path.glob('../db/img_spike/'+name+'/*.jpg')


input_neuron_data = InputNeuronData(timeDelta, taug, a, b, c, d)

model_imagepath = '../db/img/sky_best_not_converted/20170629_224036_HDR.jpg'
model_hist = returnHistogram(model_imagepath)
model_sky = returnSky(model_imagepath)


test_imagepath = '../db/img/sky_best_not_converted/20170629_224036_HDR.jpg'
test_hist = returnHistogram(test_imagepath)
test_sky = returnSky(test_imagepath)


neurons = []
neurons_sky = []
for i in range(neurons_amount):
    neurons.append(Neuron(input_neuron_data, n, 1, np.array([1])))
for i in range(neurons_amount):
    neurons_sky.append(Neuron(input_neuron_data, n, 1, np.array([1])))

hist_count_out = Neuron(input_neuron_data, n, neurons_amount, np.ones(neurons_amount))
sky_count_out = Neuron(input_neuron_data, n, neurons_amount, np.ones(neurons_amount))
hist_out = Neuron(input_neuron_data, n, 1, np.array([1]))
sky_out = Neuron(input_neuron_data, n, 1, np.array([1]))

image_list = []

name = 'sky_data'
path = Path(".")
path = path.glob('../db/img_spike/spike_four/*.jpg')

image_list_count = 4
for imagepath in path:
    image_list.append(str(imagepath))

max_probe = 200
part_t = 0
for t in range(T - 1):
    if t % 1000 == 0:
        part_t = 0
        
    for i in range(neurons_amount):
        if test_hist[i] == t:
            neurons[i].setCurrent(2, t, 0)
            print("hist "+ str(t) + " "+str(i))
        neurons[i].calc(t)
    for i in range(neurons_amount):
        if test_sky[i] == t:
            neurons_sky[i].setCurrent(2, t, 0)
            print("sky "+ str(t) + " "+str(i))
        neurons_sky[i].calc(t)
    if t > 10:
        for j in range(neurons_amount):
            if neurons[j].fired == t and model_hist[j] == t - 11:
                hist_count_out.setCurrent(1, t, j)
                print("hist fired " + str(t) + " " + str(i))
        for j in range(neurons_amount):
            if neurons_sky[j].fired == t and model_sky[j] == t - 11:
                sky_count_out.setCurrent(1, t, j)
                print("sky fired " + str(t) + " " + str(i))
    hist_count_out.calc(t)
    sky_count_out.calc(t)

    if t >970 and t < 975 and hist_count_out.u_step[t] > treshold:
        hist_out.setCurrent(1, t, 0)
    else:
        hist_out.setCurrent(0, t, 0)
    hist_out.calc(t)
    if t > 980 and t < 985 and sky_count_out.u_step[t] > treshold:
        sky_out.setCurrent(1, t, 0)
    else:
        sky_out.setCurrent(0, t, 0)
    sky_out.calc(t)





#plt.figure(figsize=(16,10))
fig, axs = plt.subplots(neurons_amount, 1, sharex=True, sharey=True)
#plt.ylabel('voltage [mV]')
#plt.title('histogram spikes')
fig.text(0.53, 0.02, 'time [ms]', ha='center')
fig.text(0.02, 0.48, 'voltage [mV]', va='center', rotation='vertical')
for i in range(neurons_amount):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].v[0:T], label = 'neuron v'+ str(i), color='blue')
    axs[i].legend(loc="upper right")
    #axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].u, label = 'neuron u'+ str(i))
#plt.xlabel('time [ms]')
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
#sky
#plt.figure(figsize=(16,10))
fig, axs = plt.subplots(neurons_amount, 1, sharex=True, sharey=True)
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel('time [ms]')
# plt.ylabel('voltage [mV]')
fig.text(0.53, 0.02, 'time [ms]', ha='center')
fig.text(0.02, 0.48, 'voltage [mV]', va='center', rotation='vertical')
#plt.title('histogram spikes')
for i in range(neurons_amount):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons_sky[i].v[0:T], label = 'neuron v'+ str(i), color='green')
    axs[i].legend(loc="upper right")
    #axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].u, label = 'neuron u'+ str(i))

# plt.title("Sky ")
#plt.figsize((16,9))
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.show()

plt.plot(model_hist, 'tab:green', label = 'model hist')
plt.plot(test_hist + 50, 'tab:blue', label = 'test hist + offset')

plt.legend(loc="upper left")
plt.show()
# plt.figure(figsize=(12, 8))
plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.v[0:T], label = 'v hist', color='blue')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.u[0:T], label = 'u step hist')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.u_step[0:T], label = 'u step')

plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.v[0:T], label = 'v_last', color='orange')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u[0:T], label = 'u_last')
#plt.grid(color='grey', linestyle=':', linewidth=1)
# plt.title('counter neuron')
# plt.legend(loc="upper left", prop={"size":20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

# plt.figure(figsize=(12,8))
plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.v[0:T], label = 'v sky', color='green')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.u[0:T], label = 'u step sky')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.u_step[0:T], label = 'u step')
plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.v[0:T], label = 'v_last', color='purple')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.u[0:T], label = 'u_last')
#plt.grid(color='grey', linestyle=':', linewidth=1)
# plt.title('counter neuron sky')
# plt.legend(loc="upper left", prop={"size":20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.v[0:T], label = 'v sky', color='green')
plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.v[0:T], label = 'v_last')
plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.v[0:T], label = 'v hist')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u[0:T], label = 'u')
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#plt.grid(color='grey', linestyle=':', linewidth=1)
#plt.title('last neuron')
#plt.legend(loc="upper left")
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

