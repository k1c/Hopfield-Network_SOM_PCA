"""
COMP 4107 Assignment #3

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 1: Hopfield Network
"""
import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from random import randint
import Hopfield

def f(x):
    return (x[1] == 1.0 or x[1] == 5.0)

def conv2binary(data):
    for i in range(len(data[0])):
        if (data[0][i] == 0):
            data[0][i] = -1
        else:
            data[0][i] = 1
    return np.asarray(data)

print("\n______________Fetching MNIST_________________")

mnist = fetch_mldata('MNIST original')

taining_zipped = zip(mnist.data[:60000], mnist.target[:60000])
testing_zipped = zip(mnist.data[60000:], mnist.target[60000:])

filter_train = filter(f, taining_zipped )
filter_test = filter(f, testing_zipped )

training_data = np.asarray([a for a in filter_train])
testing_data = np.asarray([a for a in filter_test])

print("\n_______________Hopfield_____________________")

hop = Hopfield.Network(28)

#Hopfield network can only store approximately 0.15N patterns.
n_inputs = 17
training = []
testing = []

# train network on X number of inputs
# Half inputs are 1s, and the other half is 5s
for i in range(int(n_inputs/2)):
    training.append(training_data[i*randint(1, 40)])
    testing.append(testing_data[i*randint(1, 40)])

for i in range(int(n_inputs/2), 0, -1):
    training.append(training_data[-i*randint(1, 40)])
    testing.append(testing_data[-i*randint(1, 40)])

#convert from unsigned int to signed int
training = [(np.int8(val[0]), val[1]) for val in training]
testing = [(np.int8(val[0]), val[1]) for val in testing]

#convert data to binary
training = np.asarray([val for val in map(conv2binary, training)])
testing = np.asarray([val for val in map(conv2binary, testing)])

np.random.shuffle(training)
np.random.shuffle(testing)

#train network on each image in training_set
hop.train(np.asmatrix([image[0] for image in training]))

#Classification accuracy is be measured by the number of correct classifications divided by the number of testing_images
print('\nClassification Accuracy: %.2f%%' % ((hop.classification_accuracy(training, testing)/float(len(testing))) * 100))
print("\n")


"""
Used for visually testing individual images

print([i[1] for i in training])
test_num = int(n_inputs/2)

test = training[test_num].copy()
for i in range(n_inputs):
    p = randint(0, 783)
    test[0][p] *= -1


pixels = test[0].reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()

res = hop.retrieve_pattern(test[0])
pixels = res.reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()
"""
