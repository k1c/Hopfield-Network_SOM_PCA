"""
COMP 4107 Assignment #3

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 3: Principal Component Analysis (PCA)
"""
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

random_seed = 45

# load data
# scale images to reduce # of input pixels from 2914 to 1850
# use peeple who have at least 50 images to reduce # of classes from 5794 to 12
lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.4)
raw_data = lfw_people.data
labels_ = lfw_people.target
names = lfw_people.target_names

num_faces = raw_data.shape[0]
num_features = raw_data.shape[1]
num_classes = names.shape[0]
labels = np.eye(num_classes)[labels_]

# split data into training/testing sets
raw_data_train, raw_data_test, labels_train, labels_test = train_test_split(raw_data, labels, random_state=random_seed)


# use PCA to get orthonormal data
pca = PCA()
pca.fit(raw_data_train)

pca_data = pca.transform(raw_data_train)
pca_test = pca.transform(raw_data_test)

# set up network

# layers
input_layer = num_features
hidden_layer = 100
output_layer = num_classes

# input, label placeholders
input_ = tf.placeholder(tf.float64, shape=[None, input_layer])
input_label = tf.placeholder(tf.float64, shape=[None, output_layer])

# weights and biases
w_1 = tf.Variable(np.random.rand(input_layer, hidden_layer), dtype=tf.float64)
b_1 = tf.Variable(np.random.rand(1, hidden_layer), dtype=tf.float64)
w_2 = tf.Variable(np.random.rand(hidden_layer, output_layer), dtype=tf.float64)
b_2 = tf.Variable(np.random.rand(1, output_layer), dtype=tf.float64)


# forward propagation
hidden_activation = tf.sigmoid(tf.matmul(input_, w_1) + b_1)
network_activation = tf.matmul(hidden_activation, w_2) + b_2
prediction = tf.argmax(network_activation, axis=1)

# loss function and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_label, logits=network_activation))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print("Network is training...")

'''
to train on raw data use {input_: raw_data_train, input_label: labels_train}
to test on raw data use {input_: raw_data_test, input_label: labels_test}

to train pca data use {input_: pca_data, input_label: labels_train}
to test on pca data use {input_: pca_test, input_label: labels_train}
'''

for i in range(200):
    sess.run(train, feed_dict={input_: pca_data, input_label: labels_train})

training_accuracy = np.mean(np.argmax(labels_train, axis=1) == sess.run(prediction, feed_dict={input_: pca_data, input_label: labels_train}))
test_accuracy = np.mean(np.argmax(labels_test, axis=1) == sess.run(prediction, feed_dict={input_: pca_test, input_label: labels_test}))

print("train accuracy = %.2f%%, test accuracy = %.2f%%" % (100. * training_accuracy, 100. * test_accuracy))
sess.close()
