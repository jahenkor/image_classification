import numpy as np
import operator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

#Default
max_iter = 500

def main():

    #k = 3, 3 classes
    iris_data = LoadIrisTestData()

    k = 3 #number of clusters

    testIris(iris_data, k)

    return 0

def testIris(iris_data, k):


    #convert dataframe to numpy array
    #print(iris_data[0].values)
    #print(iris_data.columns[0])

#Map down to 2 dimensions to visualize clusters
    pca = PCA(n_components=2).fit(iris_data)
    iris_2d = pca.transform(iris_data)

    col1 = iris_2d[:,0]
    col2 = iris_2d[:,1]


    iris_data = iris_data.values

    #PCA plot
    #plt.scatter(col1, col2, c='black')

    KMeans(iris_data, k)







def KMeans(dataset, k):


    X = []
    C = np.zeros((k,(dataset.shape)[1]), float)
    C_new = np.zeros((k,(dataset.shape)[1]), float)

    #clusters = np.empty(k, int)
    clusters = []



    #Intial Centroids, choose random points from data
    for i in range(k):
        index = np.random.randint(len(dataset))
        print(dataset[index])
#        X = np.array(np.array((np.copy(dataset[index]))))
        C[i] = dataset[index]

    #Remove centroid points from dataset
        dataset = np.delete(dataset, index, axis=0)

    print(clusters)

    print("Centroids: ")
    print(C)
    print(len(C))
    print("iris data")
    print(dataset)
    print(len(dataset))


    #Kmeans++
    #To-Do

    #Initial plots w/ centroid
    cluster_plot(dataset, C)

    #Assign points to centroid
    for i in range((dataset.shape)[0]):
        #print(clusters[ComputeDistanceToPoint(dataset[i],C)])
        clusters.append((ComputeDistanceToPoint(dataset[i],C),dataset[i]))

    clusters = np.array(clusters)
    print("clusters")
    print(clusters)

for i in range(max_iter):
    #Recompute centroid
   # cluster1 = 0
   # cluster2 = 0
    numclusters = np.zeros(k, int)
    print("numclusters before")
    print(numclusters)
    for j in range((clusters.shape)[0]):
        index = clusters[j][0]
        print(index)
        print("Adding")
        print(clusters[j][1])
        print(C_new[index])
        C_new[index] = C_new[index] + clusters[j][1]
        print("equal to")
        print(C_new[index])
        numclusters[index] = numclusters[index] + 1
        print("Number of clusters for %d" % index)
        print(numclusters[index])
        #if(clusters[j][0] == 0):
            #Testting cluster correctness
            #cluster1 += 1
        #else:
            #cluster2 += 1

    print("C new before comp")
    print(C_new)

    for i in range(k):
        C_new[i] = C_new[i]/numclusters[i]
    print("C New after comp")
    print(C_new)
    print(C)
    #End recompute centroid

    #Test convergence
    hasConverged = convergence(C,C_new, k)
    if(hasConverged):
        predict(clusters)


    #Replace C_old with C_new
    C = np.copy(C_new)
    #Re-visualize after centroid update
    cluster_plot(dataset, C)


    return 0

def predict(clusters):
#To Do, iterate through clusters and output index
    print("Old and New Cluster have converged")

    return clusters
def ComputeDistanceToPoint(point, centroids):

    distances = []
    i=0

    for centroid in centroids:
        distance = euclidean_distances(point.reshape(1,-1), centroid.reshape(1,-1))
        print(distance.shape)
        distances.append((i, distance))
        i+=1

    distances.sort(key = operator.itemgetter(1))
    print(distances)
    print((distances[0][0]))



#index of nearest cluster
    return (distances[0][0])

def convergence(C_old, C_new,k):
    dist = 0
    for i in range(k):
        dist += euclidean_distances(C_new[i].reshape(1,-1), C_old[i].reshape(1,-1))
    if(dist == 0):
        return True
    return False

def LoadIrisTestData():
    test_dataset = pd.read_csv("iris-cluster/1553693061_5494957_iris_new_data.txt", sep=" ", engine="python", header=None)

    return test_dataset

def LoadImageProcTestData():
    test_dataset = pd.read_csv("1553652959_123213_new_test.txt", sep=",", engine="python", header=None)

    return test_dataset

def cluster_plot(dataset, C):
    ThreeD_plot = Axes3D(plt.figure())

    ThreeD_plot.scatter(dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3])
    ThreeD_plot.scatter(C[:,0],C[:,1],C[:,2],C[:,3], marker = '*', c='r')



    plt.show()


main()
