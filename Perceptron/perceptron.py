# A 2D perceptron

# Decision boundary:
# f(x1, x2) = w1 * x1 + w2 * x2 + w3 = 0

# Classify inputs (x1, x2) as 1 if f(x1, x2) > 0, otherwise as 0.

import numpy as np
import logging
from random import randint # when training, random training samples are picked

logging.basicConfig(level=logging.INFO)

class Perceptron:
    def __init__(self):
        self.weights = np.array([1,3,1])   # initial boundary (for 2D input)
        self.weightsequence = []           # storing boundaries during training

    def predict(self, input):
        summation = np.dot(input, self.weights)   # Assume input has form (x1, x2, 1)
        if summation > 0:
            return 1
        else:
            return 0

    def train(self, training_inputs, labels):
        logging.info("Training started")
        time_step = 0
        self.weightsequence.append(self.weights)             # save boundaries
        while self._test_all(training_inputs, labels) == 0:  # train until all training samples are correctly classified

            rint = randint(0, len(training_inputs)-1)
            x = training_inputs[rint]                      # pick random input

            summation = np.dot(x, self.weights)

            # Weight update rules:
            # If dot_before < 0 but should be > 0: 
            # weights -> weights + x, so that
            # dot_after = dot_before + ||x||^2 > dot_before
            
            # If dot_before > 0 but should be <0:
            # weights -> weights - x, so that
            # dot_new = dot_before - ||x||^2 < dot_before

            if summation <= 0 and labels[rint] == 1:
                self.weights = self.weights + x       
            elif summation >= 0 and labels[rint] == 0:
                self.weights = self.weights - x
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



