from nlpPipeline1.backend import individualModules as im
import numpy as np
from scipy.misc import comb
import pickle
from sklearn.preprocessing import normalize
from nlpPipeline1.backend.plotdata import PlottingData
import os
from django.conf import settings
import json
from collections import Counter
from sklearn import metrics as metrics

#pathToData="datasets/custom2/"
#pathToPickles = "datasets/custom2/"


class pipeLine:

    def __init__(self):
        self.filePath=None
        self.picklePath=None
        self.normalizedData=None
        self.clusterCount=None
        self.labels=None
        self.centroids=list()
        self.fileDictionary=None
        self.entities=None
        self.fileNames=None
        self.absoluteFileNames=None
        self.randScore={'rand': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


    def readData(self,filePath,picklePath):
        print("Begin reading data")
        self.filePath=filePath
        self.picklePath = picklePath
        custom2Names = open(os.path.join(self.picklePath, 'plotNamesOfDocs'), 'rb')
        self.fileNames = pickle.load(custom2Names)
        custom2Pickle = open(os.path.join(self.picklePath, 'plotValuesOfDocs'), 'rb')
        total1 = pickle.load(custom2Pickle)

        self.normalizedData = normalize(total1)


    def customKmeansExecute(self):
        print("Beginning Clustering")

        (self.clusterCount, clf) = im.customKMeansComplete(self.normalizedData)

        print("Clustering done with", self.clusterCount)
        self.customKMeansProcessing(clf)

    def customKMeansProcessing(self,clf):
        self.labels = clf.getLabels(self.normalizedData)
        centroidsCustom = clf.centroids
        for each in range(len(centroidsCustom)):
            centroid = centroidsCustom[each]
            self.centroids.append(list(centroid))
        self.centroids = np.asarray(self.centroids)
        self.fileDictionary = im.getDocClustersNames(self.clusterCount, self.labels, self.fileNames)
        print("File dict generated")
        for key, val in self.fileDictionary.items():
            self.fileDictionary[key] = [os.path.join(self.filePath, file) for file in self.fileDictionary[key]]
        self.absoluteFileNames = [os.path.join(self.filePath, file) for file in self.fileNames]

    def skLearnKmeans(self):
        (self.clusterCount, clf) = im.skLearnKMeansComplete(self.normalizedData)
        self.skLearnProcessing(clf)


    def skLearnProcessing(self,clf):
        self.labels = clf.labels_

        self.centroids = (clf.cluster_centers_)

        self.fileDictionary = im.getDocClustersNames(self.clusterCount, self.labels, self.fileNames)
        print("File dict generated")
        for key, val in self.fileDictionary.items():
            self.fileDictionary[key] = [os.path.join(self.filePath, file) for file in self.fileDictionary[key]]
        self.absoluteFileNames = [os.path.join(self.filePath, file) for file in self.fileNames]


    def hybridKMeans(self):
        (typeOfAlg,self.clusterCount,clf)=im.hybridKmeans(self.normalizedData)
        if typeOfAlg == 0:
            self.customKMeansProcessing(clf)
        else:
            self.skLearnProcessing(clf)


    def birchExecute(self):
        (self.clusterCount,brc)=im.skLearnBirch(self.normalizedData)
        self.labels=brc.labels_
        self.centroids=brc.subcluster_centers_
        self.fileDictionary = im.getDocClustersNames(self.clusterCount, self.labels, self.fileNames)
        print("File dict generated")
        for key, val in self.fileDictionary.items():
            self.fileDictionary[key] = [os.path.join(self.filePath, file) for file in self.fileDictionary[key]]
        self.absoluteFileNames = [os.path.join(self.filePath, file) for file in self.fileNames]

    def getNamedEntities(self):
        self.entities = im.getNamedEntties(self.filePath, self.fileDictionary, 10)

    def getNamedEntitiesAPI(self):
        self.entities = im.getNamedEntitiesForAPI(self.filePath, self.fileDictionary)
        #print(self.entities)

    def setRandScore(self,randDict):
        self.randScore=randDict


    def sendToPlotData(self):
        pd = PlottingData()
        pd.set_filenames(self.absoluteFileNames)
        pd.set_points(self.normalizedData)
        pd.set_colors(self.labels)
        pd.set_clusters(self.clusterCount, self.fileDictionary, self.centroids)
        pd.set_named_entities(self.entities)
        pd.set_rand(self.randScore)
        pd.prepare_to_plot()

    def calculateF1Score(self,trueLabels):


        rand=metrics.adjusted_rand_score(trueLabels,self.labels)

        print("Rand score",rand)

    def rand_index_score(self,clusters, classes):

        tp_plus_fp = comb(np.bincount(clusters), 2).sum()
        tp_plus_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                 for i in set(clusters))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1=((2.0 * precision * recall) / (precision + recall))
        rand=(tp + tn) / (tp + fp + fn + tn)
        rand = round(rand, 5)
        precision = round(precision, 5)
        f1 = round(f1, 5)

        return {'rand': rand, 'precision': precision, 'recall': recall, 'f1': f1}


temp={0:[3, 7, 17, 18, 20, 28, 36, 41, 42, 39, 60, 62, 71, 72, 73, 74, 75, 77, 80, 81, 82, 83, 87, 90, 91, 92],
          1:[1, 8, 13, 14, 15, 16, 24, 31 , 33, 34, 44, 53, 65, 68, 84],
          2:[4, 5, 6, 10, 12, 19, 23, 25, 29, 32, 37, 38, 46, 48, 49, 50, 51, 52, 55, 64, 67, 70, 93],
          3:[0, 2, 9, 22, 35, 45, 66, 88, 89, 94, 40, 47, 56, 57, 58, 59, 69, 78, 86, 95, 21],
          4:[26, 54, 61, 79, 85,27, 43],
          5:[11, 30, 63, 76]}

movie_plotTrueLabels=[]

missing=[]
for key,value in temp.items():
    print(len(value))
    for i in value:
        missing.append(i)
        movie_plotTrueLabels.insert(i,key)


customTrueLabels = [0,1,2,0,0,1,1,0,1,1,3,2,3,4,2,5,0,6,1,2,2]


def run(fpath, pdir,dataSet):

    pipe=pipeLine()
    pipe.readData(fpath,pdir)
    pipe.skLearnKmeans()
    #print(pipe.fileDictionary)
    for i in range(len(pipe.labels)):
        print("index",i,">>>>>>",pipe.fileNames[i],":",pipe.labels[i])

    print(pipe.labels," len",len(pipe.labels))
    print("True labels>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")



    print("Cluster count",pipe.clusterCount)

    if dataSet == "source2":
        pass
        pipe.randScore= pipe.rand_index_score(pipe.labels, customTrueLabels)
    elif dataSet == "movie_plots":
        pipe.randScore = pipe.rand_index_score(pipe.labels, movie_plotTrueLabels)
    else:
        pipe.randScore={'rand': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    #print("rand score",score)

    pipe.getNamedEntities()
    pipe.sendToPlotData()
    #im.plotClusters(normalized,fileNames,labels,centroids,True)



def runForAPI(fpath,pdir,dataSet):
    pipe = pipeLine()
    pipe.readData(fpath, pdir)
    pipe.skLearnKmeans()
    pipe.getNamedEntitiesAPI()
    if dataSet == "source2":
        score = pipe.rand_index_score(pipe.labels, customTrueLabels)
    elif dataSet == "movie_plots":
        score = pipe.rand_index_score(pipe.labels, movie_plotTrueLabels)
    else:
        score={'rand': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    #print(pipe.entities)
    rand=score['rand']
    precision=score['precision']
    recall=score['recall']
    f1=score['f1']
    finalData=list()
    for each in pipe.fileDictionary:
        temp=dict()
        temp['files']=pipe.fileDictionary[each]
        temp['summary']=pipe.entities['summary'][each]
        finalData.append(temp)

    stats=dict()
    stats['clusterCount']=pipe.clusterCount
    stats['fileCount']=len(pipe.fileNames)
    stats['randIndex']=rand
    stats['precision']=precision
    stats['recall']=recall
    stats['f1']=f1
    return {"dict":finalData,"stats":stats}