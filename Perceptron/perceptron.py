# A perceptron with two inputs that can be trained.
# Can only learn linearly separated patterns,
# otherwise training won't stop

import numpy as np
import logging
from random import randint # when training, random training samples are picked

logging.basicConfig(level=logging.INFO)

class Perceptron:
    def __init__(self):
        self.weights = np.array([1,3,1])   # initial 'random' weights
        self.weightsequence = []           # storing weights during training

    def predict(self, input):
        summation = np.dot(input, self.weights)   # input is assumed to be in the extended form (x1, x2, 1)
        if summation > 0:
            return 1
        else:
            return 0

    def train(self, training_inputs, labels):
        logging.info("Training started")
        time_step = 0
        self.weightsequence.append(self.weights)              # add the first weights to weightsequence
        while self._test_all(training_inputs, labels) == 0:   # while the training_inputs are not all correctly predicted

            rint = randint(0, len(training_inputs)-1)
            x = training_inputs[rint]                      # pick a random input from training_inputs

            summation = np.dot(training_inputs[rint], self.weights)

            if summation <= 0 and labels[rint] == 1:
                self.weights = self.weights + x               # add x to weights if x is supposed to be on the positive side of the separating plane but is not
                self.weightsequence.append(self.weights)
            elif summation >= 0 and labels[rint] == 0:
                self.weights = self.weights - x               # subtract x from weights if x is supposed to be on the negative side of the separating plane but is not
                self.weightsequence.append(self.weights)

            time_step += 1

        logging.info("Number of steps: %d", time_step)


    def _test_all(self, training_data, labels):
        """ Return 0 if at least one sample is not correctly labeled, othwerwise return 1 """
        for i in range(len(labels)):
            summation = np.dot(training_data[i], self.weights)
            if summation <= 0 and labels[i] == 1:
                return 0
            if summation >= 0 and labels[i] == 0:
                return 0
        return 1
