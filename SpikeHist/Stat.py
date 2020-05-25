import time
import numpy as np
class Stat:

    def __init__(self, image_amount):
        self.hist_fired = 0
        self.sky_fired = 0
        self.neuron_amount = 0
        self.isLandscape = False
        self.executionTime = 0
        self.start = 0
        self.image_amount = image_amount
        self.sky_fired_tab = np.zeros((image_amount + 1))
        self.hist_fired_tab = np.zeros((image_amount + 1))


    def StartTimer(self):
        self.start = time.time()

    def EndTimer(self):
        self.executionTime = time.time() - self.start