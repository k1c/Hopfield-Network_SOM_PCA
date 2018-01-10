"""
COMP 4107 Assignment #3

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 2:
K-Means lines: 53-73
Self-Organizing Map lines: 79-92
"""
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import SOM

# get 1s and 5s
def check_digit(x):
    return x[1] in [1.0, 5.0]

print("\n______________Fetching MNIST_________________")

mnist = fetch_mldata('MNIST original')

zipped = zip(mnist.data[:60000], mnist.target[:60000])
filtered_iterable = filter(check_digit, zipped)

filtered_list = [item for item in filtered_iterable]

# data: (12163, 784), holds all image data (pixels)
# labels: (12163, ), holds all corresponding labels for the data
data = np.asarray([x[0] for x in filtered_list])
labels = np.asarray([int(x[1]) for x in filtered_list])

# PCA of data
pca = PCA(n_components=2)
pca.fit(data)

# reduce dimensions to 2D
pca_list = pca.transform(data)

# plot 1s and 5s
plt.scatter(pca_list[:, 0], pca_list[:, 1], c=labels, cmap='rainbow', s=0.5)
plt.title('Plot of 1s (purple) and 5s (red) without kmeans')
plt.xticks(())
plt.yticks(())
plt.savefig('plot_1s_and_5s_without_kmeans.png',bbox_inches='tight')
plt.clf()
#plt.show()

print("\n_______________K-Means_______________________")

# k-means
num_clusters = [2,3,4,5,6]

# run k-means for each k in num_clusters
for k in num_clusters:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    # reduce centroid dimensions to 2D as per the PCA done on data
    centroids = pca.transform(kmeans.cluster_centers_)

    # plot the clusters
    plt.scatter(pca_list[:, 0], pca_list[:, 1], c=kmeans.labels_, cmap='rainbow', s=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', s=143)
    title = "k-means with " + str(k) + " clusters"
    plt.title(title)
    print("saving " + title + ".png")
    plt.xticks(())
    plt.yticks(())
    file_name = "k-means_"+ str(k)+"_clusters"
    plt.savefig(file_name+'.png',bbox_inches='tight')
    plt.clf()
    #plt.show()

print("\n____________________SOM_______________________")

#SOM hyper-parameters
map_dim = (15,15)
learning_rate = 0.1

input_dim = 784

#train SOM: build map using training input values (1 and 5 form training)

som = SOM.Network(map_dim,input_dim,learning_rate)

som.plot_map("SOM_Pre-training_15x15")

som.train_network(data)

som.plot_map("SOM_Post-training_15x15_lr_0_point_1")
