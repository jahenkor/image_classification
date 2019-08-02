import numpy as np
import time
from sklearn.metrics import v_measure_score,mean_squared_error
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize,StandardScaler,scale, MinMaxScaler
import operator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Default
max_iter = 25
tol = 0.0001
centroidarray = []
totalSSE = []
_distance = DistanceMetric.get_metric('euclidean')

def main():

    global totalSSE


    k = 22 #max number of clusters


    clusters = [i for i in range(2,k+1,2)]
    print(clusters)



    data = LoadImageProcTestData()
    #data = LoadIrisTestData()
    for i in range(2,k+1,2):
        print("K is %d" %i)
        testIris(data, i)
        print(totalSSE)

    print(clusters)
    print(totalSSE)

    #Plot SSE over k clusters - elbow method

    plt.plot(clusters, totalSSE)
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("Image Data")
    plt.show()

    return 0

def testIris(iris_data, k):


    #Test algorithm on small dataset
    #train,test = train_test_split(iris_data, train_size = 0.07, shuffle=False)
    #iris_data = train


    #Use this to make np array
    iris_data = iris_data.values

    #Can choose array, random or kmeans++
    init = 'array'

    #Normalize data, unit norm

    #Scale down values, unit variance
    #scaler = StandardScaler()
    #scaler.fit(iris_data)
    #iris_data = scaler.transform(iris_data)

    #Scale data from 0 to 1 : Image Classification
    #scaler = MinMaxScaler()
    #iris_data = scaler.fit_transform(iris_data)
#    iris_data = scale(iris_data)



    if(init == 'array'):
        #PCA, lower dimensional space, use for visualization
        pca = PCA(n_components=k).fit(iris_data)
        #iris_data = pca.transform(iris_data)
        global centroidarray
        centroidarray = np.copy(pca.components_)



    #iris_data = normalize(iris_data, norm='l2')




    print("My Implementation of K Means")
    prediction=MyKMeans(iris_data, k,init)








def nextCentroid(intial_centroid,dataset):

    #choose centroid in dataset furthest from current centroids
    distances = np.zeros((dataset.shape[0]),float)
    for i in range(len(dataset)):
        distances[i] = ComputeDistanceToPoint(dataset[i],intial_centroid, False)
    square_distances = distances ** 2
    probability = square_distances/square_distances.sum()


    index = np.random.choice(dataset.shape[0], 1, p=probability)

    return index

#K++
def kmeansplusplus(dataset,k):

    C = np.zeros((k,(dataset.shape)[1]), float)

#Intial Centroids
    #choose one random point from data
    first_centroid_index = np.random.randint(len(dataset))
    C[0] = dataset[first_centroid_index]


    for i in range(1,k):
        index = nextCentroid(C[:i,],dataset)
        C[i] = dataset[index]



    return C

#Random
def randomCentroid(dataset,k):


    C = np.zeros((k,(dataset.shape)[1]), float)
    #Intial Centroids, choose random points from data
    for i in range(k):
        index = np.random.randint(len(dataset))
        C[i] = dataset[index]




    return C


def MyKMeans(dataset, k,init):


    C_new = np.zeros((k,(dataset.shape)[1]), float)
    C = 0
    has_converged = False

    if init == "random":
        C = randomCentroid(dataset,k)
    elif init == 'kmeansplusplus':
        C = kmeansplusplus(dataset,k)
    else:
        C = centroidarray


    #Initial plots w/ centroid
    #cluster_plot(dataset, C)

    for i in range(max_iter):

        clusters = []
        distances = []
        C_new = np.zeros((k,(dataset.shape)[1]), float)

    #Assign points to centroid
        for i in range((dataset.shape)[0]):
            distanceToPoint = ComputeDistanceToPoint(dataset[i],C,True)
            clusters.append((distanceToPoint[0],dataset[i]))
            distances.append((i,distanceToPoint[1][1]))

        clusters = np.array(clusters)

        #Recompute centroid, mean of points in cluster
        numclusters = np.zeros(k, int)
        for j in range((clusters.shape)[0]):
            index = np.copy(clusters[j][0])
            C_new[index] = C_new[index] + np.copy(clusters[j][1])

            numclusters[index] = numclusters[index] + 1

        for i in range(k):

            C_new[i] = C_new[i]/numclusters[i]
            if(numclusters[i] == 0):
                C_new[i] = ResolveEmptyCluster(dataset,distances)
        #End recompute centroid

        #Test convergence
        hasConverged = convergence(C,C_new, k)
        if(hasConverged):
            print("convergence has occured")
            break


        #Replace old C with C_new
        C = np.copy(C_new)

        #Re-visualize after centroid update
        #cluster_plot(dataset, C)



    evaluateModel(clusters,C,k)
    predictions=predict(clusters,k)

    return predictions

def ResolveEmptyCluster(dataset,distances):

    #Choose point with highest SSE from its centroid as
    #a new centroid

    print("resolving empty clusters....")
    distances.sort(key = operator.itemgetter(1))

    distances = np.array(distances)


    #index of furthest element to centroid
    index = distances[-1][0]

    C = dataset[index]
    return C



def evaluateModel(clusters, C, k):

    #SumOfSquaredErrors
    SSE = 0
    testSSE = 0
    error_matrix = 0
    for i in range((clusters.shape)[0]):
        index = clusters[i][0]
        error_matrix += (C[index] - clusters[i][1])**2


    for i in range(len(error_matrix)):
        SSE += error_matrix[i]

    totalSSE.append(SSE)

    return SSE

def predict(clusters,k):

    #Predict cluster(s) of a dataset
    print("Prediction")

    predictions = np.zeros((clusters.shape)[0],int)

    for i in range((clusters.shape)[0]):
        predictions[i] = clusters[i][0] + 1

    print(predictions)

    with open('irisout.txt','w') as file:
        for i in range(len(predictions)):
            file.write("%s\n" % predictions[i])
        file.flush()

    return predictions

def ComputeDistanceToPoint(point, centroids, index):

    distances = []
    i=0

    for centroid in centroids:
        distance = _distance.pairwise(point.reshape(1,-1), centroid.reshape(1,-1))
        distances.append((i, distance))
        i+=1


    if(index == True):
        distances.sort(key = operator.itemgetter(1))


    if(index == False):
        return (distances[0][1])

    #index of nearest point, and furthest point to centroid
    return (distances[0][0],distances[1])

#Check if old and new clusters are within tolerance
def convergence(C_old, C_new,k):
    dist = 0
    for i in range(k):
        dist += _distance.pairwise(C_new[i].reshape(1,-1), C_old[i].reshape(1,-1))
    if(dist < tol):
        return True
    return False

#Data Input
def LoadIrisTestData():
    test_dataset = pd.read_csv("iris-cluster/1553693061_5494957_iris_new_data.txt", sep=" ", engine="python", header=None)

    return test_dataset

def LoadImageProcTestData():
    test_dataset = pd.read_csv("1553652959_123213_new_test.txt", sep=",", engine="python", header=None)

    return test_dataset

#Plot clusters, use PCA for 2D plot
def cluster_plot(dataset, C):
    #ThreeD_plot = Axes3D(plt.figure())

    plt.scatter(dataset[:,0], dataset[:,1])#,dataset[:,2], dataset[:,3])
    plt.scatter(C[:,0],C[:,1],marker='*',c='r')#,C[:,2],C[:,3], marker = '*', c='r')



    plt.show()










main()
