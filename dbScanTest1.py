from sklearn.cluster import DBSCAN
from sklearn import metrics
import pickle
import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import normalize
from operator import itemgetter
import individualModules as im


#custom2Pickle=open("/media/sharathbhragav/New Volume/redditPosts/pickles/plotValuesOfDocs","rb")
custom2Pickle=open("datasets/custom2/plotValuesOfDocs","rb")
total1=pickle.load(custom2Pickle)
normalized=normalize(total1)


custom2Names=open("/media/sharathbhragav/New Volume/redditPosts/pickles/plotNamesOfDocs","rb")
fileNames=pickle.load(custom2Names)



silhoutteScores = {}
thresholdValues={}
for i in range(2,100):
    brc = Birch(branching_factor=50, n_clusters=None, threshold=0.01*i,compute_labels=True)

    labels=brc.fit_predict(total1)
    print(len(labels))
    try:
        silhouette_avg = metrics.silhouette_score(total1, labels)
        clusterNumber=len(set(labels))
        silhoutteScores[clusterNumber] = silhouette_avg
        thresholdValues[clusterNumber] =i*0.01

    except:
        continue
sortedSil = sorted(silhoutteScores.items(), key=itemgetter(1))
selectedClusterNumber = sortedSil[-1][0]
print("selected number of clusters=", selectedClusterNumber)
brc = Birch(branching_factor=50, n_clusters=None, threshold=thresholdValues[selectedClusterNumber],compute_labels=True)
labels=brc.fit_predict(total1)
print(labels)
centroids=brc.subcluster_centers_
print(centroids)

im.plotClusters(total1,fileNames,labels,centroids)