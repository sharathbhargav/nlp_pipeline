"""
Goal:
using sklearn implement end to end pipeline
input:Path to set of documents or handles of bunch of documents
output: clusters of documents

process: obtain document vectors -> use PCA to reduce dimensionality -> normalize the vectors using z normalization -> pass the vectors to kmeans,kmeans++, spherical kmeans for clusters 2-20 ->
    obtain silhouette avg values and get best cluster number -> generate clusters with any of the above methods
"""

from gensim.models import Word2Vec
import individualModules as im
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import pickle
from sklearn.decomposition import PCA
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from operator import itemgetter
from algorithms import kMeans as kmeans
from random import randint

"""
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)

style.use('ggplot')

total1=[]

fileNames=[]


# get document vactors with PCA done and dimension of 2
custom2Pickle=open("datasets/custom2/plotValuesofDocs","rb")
total1=pickle.load(custom2Pickle)
custom2Names=open("datasets/custom2/plotNamesofDocs","rb")
fileNames=pickle.load(custom2Names)

"""
"""

pathSuffix=["b","e","p","s","t"]
for each in pathSuffix:
    pickleFile1=open("datasets/bbc/plotValue"+each,"rb")
    plotValue=pickle.load(pickleFile1)
    pickleFile2=open("datasets/bbc/plotName"+each,"rb")
    plotName=pickle.load(pickleFile2)
    for value in plotValue:
        total1.append(value)
    for name in plotName:
        fileNames.append(name)



#print(normalized)
"""
"""
normalized=normalize(total1)
colors = 100 * ["r", "g", "b", "c", "k"]
"""


colors = 100 * ["r", "g", "b", "c", "k"]




def getOptimalClustersSilhoutte(data,custom=False):
    silhoutteScores = {}
    rotationStored = {}
    if custom :
        for clusterKmeansNumber in range(2, 20):
            try:
                clf = kmeans.K_Means(clusterKmeansNumber, tolerance=0.00001, max_iterations=800)
                rotation = randamozieSeed(data, clusterKmeansNumber)
                clf.fit(data, spherical=True, rotationArray=rotation)
                labels = clf.getLabels(data)
                silhouette_avg = silhouette_score(data, labels)
                silhoutteScores[clusterKmeansNumber] = silhouette_avg
                rotationStored[clusterKmeansNumber] = rotation
                #print(clusterKmeansNumber,">>>>>>>",rotation)
            except:
                continue
                #print(clusterKmeansNumber," chucked")
    else:
        for clusterKmeansNumber in range(2, 20):
            clf = KMeans(n_clusters=clusterKmeansNumber)
            labels = clf.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhoutteScores[clusterKmeansNumber] = silhouette_avg

    sortedSil = sorted(silhoutteScores.items(), key=itemgetter(1))
    selectedClusterNumber = sortedSil[-1][0]
    print("selected number of clusters=", selectedClusterNumber)
    if custom:
        return (selectedClusterNumber,rotationStored[selectedClusterNumber])
    else:
        return selectedClusterNumber



def skLearnKMeansComplete(data,fileNames,plot=False):
    selectedClusterNumber=getOptimalClustersSilhoutte(data,False)
    clf=KMeans(n_clusters=selectedClusterNumber)
    clf.fit(data)
    centroidsKmeans=clf.cluster_centers_
    labelsKmeans=clf.labels_

    if plot:
        i=0
        for k in data:
            xy=(k[0],k[1])
            plt.scatter(k[0],k[1],color=colors[labelsKmeans[i]],marker="o",s=25,linewidths=5)
            plt.annotate(fileNames[i],xy)
            i+=1
        plt.scatter(centroidsKmeans[:,0],centroidsKmeans[:,1],marker='x',s=150,linewidths=5)
        plt.show()

    return (selectedClusterNumber,clf)



def randamozieSeed(data,k):
    outputSeed=[]
    for randomNumberIter in range(k):
        random=randint(0,len(data))
        outputSeed.append(data[random])
    return outputSeed


def customKMeansComplete(data,fileNames,plot=False):
    (selectedClusterNumber,rotation) = getOptimalClustersSilhoutte(data,True)


    clf = kmeans.K_Means(selectedClusterNumber, tolerance=0.00001, max_iterations=800)
    #print(">>>>>>>>>>>>>>>>>final rot",rotationStored[selectedClusterNumber])
    clf.fit(data, spherical=True, rotationArray=rotation)
    classifications=clf.classifications
    centroids=clf.centroids

    if plot:
        count=0
        for centroid in centroids:
            plt.scatter(centroids[centroid][0], centroids[centroid][1], marker="o", color=colors[count], s=100,
                        linewidths=5)
            count = count + 1

        for classification in classifications:
            color = colors[classification]
            if len(classifications[classification]) > 0:
                for featureSet in classifications[classification]:
                    plt.scatter(featureSet[0], featureSet[1], marker="x", color=color, s=100, linewidths=5)

        i = 0
        #print(len(fileNames))
        #print(len(clf.getLabels(normalized)))
        #print(fileNamesClusters)

        for k in data:
            xy = (k[0], k[1])

            plt.annotate(fileNames[i], xy)
            i += 1

        plt.show()


    return (selectedClusterNumber,clf)


def getDocClustersNames(clusterCount,labels,fileNames):
    fileNamesClusters = {}
    labelsOfDocs = labels
    for clusterNumber in range(clusterCount):
        singleClusterDocs = []
        for getDocData in range(len(labels)):
            if labelsOfDocs[getDocData] == clusterNumber:
                singleClusterDocs.append(fileNames[getDocData])
        fileNamesClusters[clusterNumber] = singleClusterDocs

    return fileNamesClusters



#skLearnKMeansComplete()

#customKMeansComplete()