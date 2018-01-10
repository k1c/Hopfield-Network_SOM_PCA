"""
COMP 4107 Assignment #3

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466
"""


import numpy as np
from math import inf
from random import shuffle
import matplotlib.pyplot as plt

class Network:

    def __init__(self, map_dim, input_dim, learning_rate):

        self.weights = np.random.random((map_dim[0],map_dim[1],input_dim))

        self.x, self.y = map_dim

        #initialize neibhourhood radius
        self.initial_radius = max(map_dim[0],map_dim[1]) / 2
        self.initial_learning_rate = learning_rate


    def get_bmu_index(self, input_):
        min_distance = inf
        bmu_index = [None,None]

        #loop through 2D map
        for x in range(self.x):
            for y in range(self.y):
                dist = np.linalg.norm(input_ - self.weights[x][y])   #calculate eucledian distance
                if dist < min_distance:
                    min_distance = dist
                    bmu_index = [x,y]
        return bmu_index

    def update_weights(self, input_, bmu_index, learning_rate, radius):

        #loop through 2D map
        for x in range(self.x):
            for y in range(self.y):
                neuron_weight = self.weights[x, y, :]

                #check if neuron_weight 2D coordinates are within the neibhourhood radius
                neuron_dist = np.sum((np.asarray([x, y]) - bmu_index) ** 2)   #distance between two points on a 2D graph without the sqaure root (too expensive)
                if neuron_dist <= radius**2:
                    #use a gaussian to find the influence of the input on the weight
                    neighbor_scaling = np.exp(-neuron_dist/(2 * radius**2))
                    new_weight = neuron_weight + ((learning_rate * neighbor_scaling) * (input_ - neuron_weight))
                    self.weights[x, y, :] = new_weight

    def train_network(self, training_data):
        shuffle(training_data)   #we shuffle our data to simulate choosing a random vector
        training_size = len(training_data)
        time_constant = training_size / np.log(self.initial_radius)
        lr = self.initial_learning_rate
        rad = self.initial_radius

        for i in range(training_size):
            if i%1000 == 0:
                print("data training at: ", i)
            bmu_index = self.get_bmu_index(training_data[i])
            self.update_weights(training_data[i], bmu_index, lr, rad)

            #decay learning_rate and radius
            lr = self.initial_learning_rate * np.exp(-i / training_size)
            rad = self.initial_radius * np.exp(-i / time_constant)


    def test_network(self, testing_images):

        for test_image in testing_images:
            #render test image and return bmu
            pixels = (test_image * 255.0).reshape((28, 28))
            plt.imshow(pixels, cmap='gray')

            bmu_index = self.get_bmu_index(test_image)
            print(bmu_index)
            plt.show()
            break

    def plot_map(self, file_name):
        fig = plt.figure(figsize=(self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                pixels = (self.weights[x][y]*255.0).reshape((28, 28))
                fig.add_subplot(self.x, self.y, (self.y*x) + y + 1)
                plt.axis('off')
                plt.imshow(pixels, cmap='gray')
        print("saving", file_name)
        fig.savefig(file_name)
        plt.close(fig)
