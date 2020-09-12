import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from Neuron import *
from InitData import *

neurons_amount = 4
image_list = []

path = Path(".")
path = path.glob('../db/img_spike/spike_new/*.jpg')

image_list_count = 0
for imagepath in path:
    image_list_count += 1
    image_list.append(str(imagepath))


treshold = (-14 + neurons_amount * d) * 0.1
out_treshold = (-14 + image_list_count * d) * 0.2

max_probe = 200
part_t = 0
iteration = 0

hist_count_offset = 500
sky_count_offset = 50

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
    sky_hist = np.empty((neurons_amount, 1))
    for col in b.T:
        average = np.average(col)
        sky_hist[i] = math.floor(average)
        i += 1
    return sky_hist


# name = 'model'
# path = Path(".")
# path = path.glob('../db/img_spike/'+name+'/*.jpg')


input_neuron_data = InputNeuronData(timeDelta, taug, a, b, c, d)

# model_imagepath = '../db/img_spike/model/IMG_20200321_182525.jpg'
# model_hist = returnHistogram(model_imagepath)
# model_sky = returnSky(model_imagepath)

test_imagepath = image_list[0]
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

out_positive = Neuron(input_neuron_data, n, 1, np.array([1]))
out_negative = Neuron(input_neuron_data, n, 1, np.array([1]))


for t in range(T - 1):

    if t % single_iteration_time == 0 and t < T - end_spike_time:
        part_t = 0
        model_hist = returnHistogram(image_list[iteration])
        model_sky = returnSky(image_list[iteration])
        iteration += 1
    for i in range(0,neurons_amount):
        if test_hist[i] == part_t:
            neurons[i].setCurrent(2, t, 0)
            #print("hist " + str(t) + " " + str(i))
        neurons[i].calc(t)
    for i in range(0,neurons_amount):
        if test_sky[i] == part_t:
            neurons_sky[i].setCurrent(2, t, 0)
            #print("sky " + str(t) + " " + str(i))
        neurons_sky[i].calc(t)
    #check fired
    if t > 10 and t < T - end_spike_time:
        part_t_back = part_t - 11
        for j in range(neurons_amount):
            if neurons[j].fired == t and model_hist[j] > part_t_back - hist_count_offset and model_hist[j] < part_t_back + hist_count_offset:
                hist_count_out.setCurrent(1, t, j)
                print("hist fired " + str(iteration) + " " + str(t) + " " + str(j))
        for j in range(neurons_amount):
            if neurons_sky[j].fired == t and model_sky[j] > part_t_back - sky_count_offset and model_sky[j] < part_t_back + sky_count_offset:
                sky_count_out.setCurrent(1, t, j)
                print("sky fired " + str(iteration) + " " + str(t) + " " + str(j))
    hist_count_out.calc(t)
    sky_count_out.calc(t)
    #last neuron
    if part_t > single_iteration_time - 140 and part_t < single_iteration_time - 135 and hist_count_out.u_step[t] > treshold:
        hist_out.setCurrent(1, t, 0)
    else:
        hist_out.setCurrent(0, t, 0)
    hist_out.calc(t)

    if part_t > single_iteration_time - 30 and part_t < single_iteration_time - 25 and sky_count_out.u_step[t] > treshold:
        sky_out.setCurrent(1, t, 0)
    else:
        sky_out.setCurrent(0, t, 0)
    sky_out.calc(t)

    if part_t > single_iteration_time - 10 and part_t < single_iteration_time - 5:
        sky_count_out.u_step[t + 1] = sky_count_out.u[t]
        hist_count_out.u_step[t + 1] = hist_count_out.u[t]

    if t > T - 320 and t < T - 315:
        if sky_out.u_step[t] > out_treshold and hist_out.u_step[t] > out_treshold:
            print("last neuron fired")
            out_positive.setCurrent(1, t, 0)
        else:
            print("last neuron not fired")
            out_negative.setCurrent(1, t, 0)
    else:
        out_positive.setCurrent(0, t, 0)
        out_negative.setCurrent(0, t, 0)
    out_positive.calc(t)
    out_negative.calc(t)

    part_t += 1


#plt.figure(figsize=(16, 10))
fig, axs = plt.subplots(neurons_amount, sharex=True, sharey=True)
#plt.ylabel('voltage [mV]')
#plt.title('histogram spikes')
fig.text(0.53, 0.01, 'time [ms]', ha='center')
fig.text(0.02, 0.48, 'voltage [mV]', va='center', rotation='vertical')
offset = 0
for i in range(neurons_amount):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].v, label='neuron v' + str(i), color="green")
    axs[i].legend(loc="upper right")

    # axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].u, label = 'neuron u'+ str(i))
#plt.xlabel('time [ms]')
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

#plt.figure(figsize=(16, 10))
fig, axs = plt.subplots(neurons_amount, sharex=True, sharey=True)
#plt.ylabel('voltage [mV]')
#plt.title('histogram spikes')
fig.text(0.53, 0.01, 'time [ms]', ha='center')
fig.text(0.02, 0.48, 'voltage [mV]', va='center', rotation='vertical')
for i in range(neurons_amount):
    axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons_sky[i].v, label='neuron v' + str(i), color="blue")
    axs[i].legend(loc="upper right")
    # axs[i].plot(np.arange(0, timeDelta * T, timeDelta), neurons[i].u, label = 'neuron u'+ str(i))
#plt.xlabel('time [ms]')
# plt.figsize((16,9))
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

#show hist plot
# offset = 0
# offset_step = 40
# for imgg in image_list:
#     hist = returnHistogram(imgg)
#     plt.plot(hist + offset, label=imgg)
#     offset += offset_step
# # plt.plot(model_hist, 'tab:green', label='model hist')
# plt.plot(test_hist + offset, 'tab:blue', label='test hist + offset')
# plt.legend(loc="upper left")
# plt.title('histogram + test models')
# plt.show()

#show sky plot
# offset = 0
# offset_step = 40
# for imgg in image_list:
#     sky = returnSky(imgg)
#     plt.plot(sky + offset, label=imgg)
#     offset += offset_step
# # plt.plot(model_hist, 'tab:green', label='model hist')
# plt.plot(test_sky + offset, 'tab:blue', label='test sky + offset')
# plt.legend(loc="upper left")
# plt.title('sky + test models')
# plt.show()

treshold_np = np.ones(T) * treshold

#plt.figure(figsize=(12, 8))

plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.v, label='v hist', color="green")
#plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.u, label='u step hist')
#plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_count_out.u_step, label='u step')

plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.v, label='v_last', color="purple", linewidth=3)
#plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u, label='u_last')

#plt.plot(np.arange(0, timeDelta * T, timeDelta), treshold_np, label='treshold')
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#plt.grid(color='grey', linestyle=':', linewidth=1)
#plt.title('counter neuron hist')
#plt.legend(loc="upper left", prop={"size": 20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

# plt.figure(figsize=(12, 8))
plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.v, label='v sky', color='blue')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.u, label='u step sky')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_count_out.u_step, label='u step')

plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.v, label='v_last', color='orange', linewidth=3)
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.u, label='u_last')


#plt.plot(np.arange(0, timeDelta * T, timeDelta), treshold_np, label='treshold')
#plt.grid(color='grey', linestyle=':', linewidth=1)
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#plt.title('counter neuron sky')
#plt.legend(loc="upper left", prop={"size": 20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.v, label='hist_v_last', color='purple')
#plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u, label='hist_u_last')
#plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u_step, label='hist_u_step_last')

plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.v, label='sky_v_last', color='orange')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.u, label='sky_u_last')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), sky_out.u_step, label='sky_u_step_last')

plt.plot(np.arange(0, timeDelta * T, timeDelta), out_positive.v, label='v_positive', color='lime', linewidth=3)
# plt.plot(np.arange(0, timeDelta * T, timeDelta), out_positive.u, label='u_positive')

plt.plot(np.arange(0, timeDelta * T, timeDelta), out_negative.v, label='v_negative', color='red')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), out_negative.u, label='u_negative')
# out_treshold_np = np.ones(T) * out_treshold
# plt.plot(np.arange(0, timeDelta * T, timeDelta), out_treshold_np, label='treshold')
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#plt.grid(color='grey', linestyle=':', linewidth=1)
#plt.title('out neuron')
#plt.legend(loc="upper left", prop={"size": 20})
plt.xlabel('time [ms]')
plt.ylabel('voltage [mV]')
# plt.figsize((16,9))
plt.show()

# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.v, label='v hist')
# plt.plot(np.arange(0, timeDelta * T, timeDelta), hist_out.u, label='u')
#
# plt.grid(color='grey', linestyle=':', linewidth=1)
# plt.title('last neuron')
# plt.legend(loc="upper left")
# plt.xlabel('time [ms]')
# plt.ylabel('voltage [mV]')
# # plt.figsize((16,9))
# plt.show()

