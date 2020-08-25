import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from Neuron import *
#from InitData import *
from Stat import *

neurons_amount = 32
image_list = []




#name_folder = "spike_small_s"
name_folder = "img8_col2"
path_folder = '../db/img_spike/' + name_folder + '/model/*.jpg'
path = Path(".")
path = path.glob(path_folder)

image_list_count = 0
for imagepath in path:
    image_list_count += 1
    image_list.append(str(imagepath))


scaler = 1

height = 1080 / scaler
width = 40 / scaler

height = int(height)
width = int(width)

taug = 10.0
frate = 0.002 #[ms^(-1)]

single_iteration_time = 1000
end_spike_time = 200
n = image_list_count * single_iteration_time + end_spike_time
timeDelta = 0.5  # 5ms
a = 0.02
b = 0.2
d = 8
c = -65

T = n
log_version = 0.5
#hist
treshold = (-14 + neurons_amount * d) * 0.3
out_treshold = (-14 + image_list_count * d) * 0.6
hist_count_offset = 80

#sky
treshold_sky = (-14 + neurons_amount * d) * 0.2
out_sky_treshold = (-14 + image_list_count * d) * 0.5
sky_count_offset = 30

max_probe = 200


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


sky_data = np.zeros((image_list_count, neurons_amount, 1), dtype=float)
hist_data = np.zeros((image_list_count, neurons_amount, 1), dtype=float)

for i in range(0, image_list_count):
    sky_data[i] = returnSky(image_list[i])
    hist_data[i] = returnHistogram(image_list[i])

print("hist and sky prepared")
# name = 'model'
# path = Path(".")
# path = path.glob('../db/img_spike/'+name+'/*.jpg')


input_neuron_data = InputNeuronData(timeDelta, taug, a, b, c, d)

# model_imagepath = '../db/img_spike/model/IMG_20200321_182525.jpg'
# model_hist = returnHistogram(model_imagepath)
# model_sky = returnSky(model_imagepath)

# test_imagepath = '../db/img_spike/test/IMG_20200321_182525.jpg'
# test_hist = returnHistogram(test_imagepath)
# test_sky = returnSky(test_imagepath)


def checkLandscape(path):
    # test_imagepath = path
    test_hist = returnHistogram(path)
    test_sky = returnSky(path)

    part_t = 0
    iteration = 0


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

    isLandscape = False
    stat = Stat(image_list_count)
    stat.neuron_amount = neurons_amount
    stat.StartTimer()
    for t in range(T - 1):

        if t % single_iteration_time == 0 and t < T - end_spike_time:
            part_t = 0
            # model_hist = returnHistogram(image_list[iteration])
            # model_sky = returnSky(image_list[iteration])
            model_hist = hist_data[iteration]
            model_sky = sky_data[iteration]
            # print("iteration: " + str(iteration))
            iteration += 1
        for i in range(0, neurons_amount):
            if test_hist[i] == part_t:
                neurons[i].setCurrent(2, t, 0)
                stat.hist_fired += 1
                stat.hist_fired_tab[iteration] += 1
                #print("hist " + str(t) + " " + str(i))
            neurons[i].calc(t)
        for i in range(0, neurons_amount):
            if test_sky[i] == part_t:
                neurons_sky[i].setCurrent(2, t, 0)
                stat.sky_fired += 1
                stat.sky_fired_tab[iteration] += 1
                #print("sky " + str(t) + " " + str(i))
            neurons_sky[i].calc(t)
        #check fired
        if t > 10 and t < T - end_spike_time:
            part_t_back = part_t - 11
            for j in range(neurons_amount):
                if neurons[j].fired == t and model_hist[j] > part_t_back - hist_count_offset and model_hist[j] < part_t_back + hist_count_offset:
                    hist_count_out.setCurrent(1, t, j)

                    # print("hist fired " + str(iteration) + " " + str(t) + " " + str(j))
            for j in range(neurons_amount):
                if neurons_sky[j].fired == t and model_sky[j] > part_t_back - sky_count_offset and model_sky[j] < part_t_back + sky_count_offset:
                    sky_count_out.setCurrent(1, t, j)

                    # print("sky fired " + str(iteration) + " " + str(t) + " " + str(j))
        hist_count_out.calc(t)
        sky_count_out.calc(t)
        #last neuron
        if part_t > single_iteration_time - 50 and part_t < single_iteration_time - 45 and hist_count_out.u_step[t] > treshold:
            hist_out.setCurrent(1, t, 0)
        else:
            hist_out.setCurrent(0, t, 0)
        hist_out.calc(t)

        if part_t > single_iteration_time - 30 and part_t < single_iteration_time - 25 and sky_count_out.u_step[t] > treshold_sky:
            sky_out.setCurrent(1, t, 0)
        else:
            sky_out.setCurrent(0, t, 0)
        sky_out.calc(t)

        if part_t > single_iteration_time - 10 and part_t < single_iteration_time - 5:
            sky_count_out.u_step[t + 1] = sky_count_out.u[t]
            hist_count_out.u_step[t + 1] = hist_count_out.u[t]

        if t > T - 20 and t < T - 15:
            if T - 19 == t:
                print(" HIST u_step: " + str(hist_out.u_step[t]) + " SKY u_step: " +str(sky_out.u_step[t]))
            if sky_out.u_step[t] > out_sky_treshold and hist_out.u_step[t] > out_treshold:
                # print("last neuron fired")
                stat.isLandscape = True
                out_positive.setCurrent(1, t, 0)
            else:
                # print("last neuron not fired")
                out_negative.setCurrent(1, t, 0)
        else:
            out_positive.setCurrent(0, t, 0)
            out_negative.setCurrent(0, t, 0)
        out_positive.calc(t)
        out_negative.calc(t)

        part_t += 1
    stat.EndTimer()
    return stat

import random


def generate(amount):
    passs = ""
    chars = "abcdefghijklmnopqrstuwvxyzABCDEFGHIJKLMNOPRSTUVWXYZ1234567890"
    for i in range(0, int(amount)):
        passs += chars[random.randint(0,len(chars)-1)]
    return passs

confusionMatrix = np.zeros((2,2))

import os
from contextlib import redirect_stdout
import sys
stdout_fd = sys.stdout.fileno()
output_file = 'stat_' + str(neurons_amount) + "_" + generate(5) + ".txt"
print("log start to file: " + output_file)
with open(output_file, 'w') as f, redirect_stdout(f):
    #print('redirected to a file')
    #os.write(stdout_fd, b'not redirected')
    #os.system('echo this also is not redirected')
    print("log version: "+str(log_version))
    print("neurons amount: " + str(neurons_amount))
    print("model img amount: " + str(image_list_count))
    print("treshold hist: " + str(treshold))
    print("treshold sky: " + str(treshold_sky))
    print("out treshold: " + str(out_treshold))
    print("out sky treshold: " + str(out_sky_treshold))

    print("hist_count_offset: " + str(hist_count_offset))
    print("sky_count_offset: " + str(sky_count_offset))
    path_folder = '../db/img_spike/' + name_folder + '/true/*.jpg'
    path = Path(".")
    path = path.glob(path_folder)

    image_list_count_true = 0
    totalTime = 0.0
    print("true start")
    for imagepath in path:
        stat = checkLandscape(str(imagepath))
        if stat.isLandscape:
            confusionMatrix[0, 0] += 1
        print("True: " + str(imagepath) + " isLandscape: " + str(stat.isLandscape) + " time: " + str(stat.executionTime))
        print("hist fired: " + str(stat.hist_fired) + " sky fired: " + str(stat.sky_fired))
        # for i in range(image_list_count):
        #     print(str(i) + " hist_fired_tab: " + str(stat.hist_fired_tab[i]) + " sky_fired_tab: " + str(stat.sky_fired_tab[i]))
        image_list_count_true += 1
        totalTime += stat.executionTime
        os.system(str(True))
    path_folder = '../db/img_spike/' + name_folder + '/false/*.jpg'
    path = Path(".")
    path = path.glob(path_folder)
    confusionMatrix[0, 1] = image_list_count_true - confusionMatrix[0, 0]
    print("false start")
    image_list_count_false = 0
    for imagepath in path:
        stat = checkLandscape(str(imagepath))
        if not stat.isLandscape:
            confusionMatrix[1, 1] += 1
        print("False: " + str(imagepath) + " isLandscape: " + str(stat.isLandscape) + " time: " + str(stat.executionTime))
        print("hist fired: " + str(stat.hist_fired) + " sky fired: " + str(stat.sky_fired))
        image_list_count_false += 1
        totalTime += stat.executionTime
        os.system(str(False))
    confusionMatrix[1, 0] = image_list_count_false - confusionMatrix[1, 1]
    print(confusionMatrix)
    print("Total Time: " + str(totalTime))
print(confusionMatrix)
print('log end ' + str(totalTime))