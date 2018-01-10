"""
COMP 4107 Assignment #3

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 1: Hopfield Network
"""
import numpy as np
from math import inf

class Network:

    def __init__ (self, image_dim):
        self.n_neurons = image_dim * image_dim
        self.weights = np.zeros([self.n_neurons, self.n_neurons])

    # hebbian training
    def train(self, dataset):
        self.weights = (dataset.T.dot(dataset) - (np.identity(len(dataset[0])) * len(dataset)))

    # calculate the output of a neuron
    def neuron_output(self, neuron, noisey_pattern):
        # output of a neuron is the dot product of the weight row of that neuron with the input vector
        activation = self.weights[neuron].dot(np.transpose(noisey_pattern))

        return 1.0 if activation > 0.0 else -1.0

    # check stability of network and reconstruct the noisey input as we go along
    # async update
    def stabalize(self, neuron_permutation, noisey_input):
        changed = False
        result = noisey_input.copy()

        # check every pixel in noisey_input and the corresponding neuron output
        # if they aren't the same, fix the noisey input
        # this transforms it towards a stable pattern
        for n in neuron_permutation:
            neuron_activation = self.neuron_output(n, result)

            if neuron_activation != result[n]:
                result[n] = neuron_activation
                changed = True

        return (changed, result)

    # reconstruct a pattern
    def retrieve_pattern(self, noisey_input):
        result = noisey_input.copy()

        while True:
            neuron_permutation = list(range(self.n_neurons))
            np.random.shuffle(neuron_permutation)

            changed, result = self.stabalize(neuron_permutation, result)

            if not changed:
                return result

    def get_winning_label(self, testing_image, training_set):
        min_distance = inf
        training_images = [image[0] for image in training_set]
        training_labels = [label[1] for label in training_set]

        training_dataset = [(image, label) for image, label in zip(training_images, training_labels)]

        for (training_image, training_label) in training_dataset:
            dist = np.linalg.norm(testing_image - training_image)   #calculate eucledian distance
            if dist < min_distance:
                min_distance = dist
                winning_label = training_label
        return winning_label

    def classification_accuracy(self, training_set, testing_set):
        score = 0
        testing_images = [image[0] for image in testing_set]
        testing_labels = [label[1] for label in testing_set]

        testing_dataset = [(image, label) for image, label in zip(testing_images, testing_labels)]

        for (testing_image,testing_label) in testing_dataset:
            winning_label = self.get_winning_label(self.retrieve_pattern(testing_image),training_set)
            if winning_label == testing_label:
                score+=1
        return score
