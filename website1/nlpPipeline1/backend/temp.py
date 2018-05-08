from gensim.models.keyedvectors import KeyedVectors
import nlpPipeline1.backend.individualModules as im
import numpy as np
from nltk.corpus import stopwords
import math
import pickle
from sklearn.preprocessing import normalize


#pathToData="/media/sharathbhragav/New Volume/redditPosts/hot/"
#pathToPickles="/media/sharathbhragav/New Volume/redditPosts/pickles/"

pathToData="datasets/custom2/"
pathToPickles = "datasets/custom2/"






def run():
    custom2Names=open(pathToPickles+"plotNamesOfDocs","rb")
    fileNames=pickle.load(custom2Names)
    custom2Pickle=open(pathToPickles+"plotValuesOfDocs","rb")
    total1=pickle.load(custom2Pickle)


    normalized=normalize(total1)


    (clusterCount,clf)=im.customKMeansComplete(normalized)


    #labels=clf.labels_
    labels=clf.getLabels(normalized)

    centroids=[]
    centroidsCustom=clf.centroids
    for each in range(len(centroidsCustom)):
        centroid=centroidsCustom[each]
        centroids.append(list(centroid))
    centroids=np.asarray(centroids)
    fileNameDictionary=im.getDocClustersNames(clusterCount,labels,fileNames)



    im.getNamedEntties(pathToData,fileNameDictionary,10)

    im.plotClusters(normalized,fileNames,labels,centroids,True)

