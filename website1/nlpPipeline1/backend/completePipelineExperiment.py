from nlpPipeline1.backend import individualModules as im
import numpy as np
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
        self.labels=clf.getLabels(self.normalizedData)
        centroidsCustom = clf.centroids
        for each in range(len(centroidsCustom)):
            centroid = centroidsCustom[each]
            self.centroids.append(list(centroid))
        self.centroids=np.asarray(self.centroids)
        self.fileDictionary=im.getDocClustersNames(self.clusterCount,self.labels,self.fileNames)
        print("File dict generated")
        for key, val in self.fileDictionary.items():
            self.fileDictionary[key] = [os.path.join(self.filePath, file) for file in self.fileDictionary[key]]
        self.absoluteFileNames=[os.path.join(self.filePath, file) for file in self.fileNames]

    def skLearnKmeans(self):
        (self.clusterCount, clf) = im.skLearnKMeansComplete(self.normalizedData)
        self.labels=clf.labels_

        self.centroids=(clf.cluster_centers_)

        self.fileDictionary = im.getDocClustersNames(self.clusterCount, self.labels, self.fileNames)
        print("File dict generated")
        for key, val in self.fileDictionary.items():
            self.fileDictionary[key] = [os.path.join(self.filePath, file) for file in self.fileDictionary[key]]
        self.absoluteFileNames = [os.path.join(self.filePath, file) for file in self.fileNames]

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


    def sendToPlotData(self):
        pd = PlottingData()
        pd.set_filenames(self.absoluteFileNames)
        pd.set_points(self.normalizedData)
        pd.set_colors(self.labels)
        pd.set_clusters(self.clusterCount, self.fileDictionary, self.centroids)
        pd.set_named_entities(self.entities)
        pd.prepare_to_plot()

    def calculateF1Score(self,trueLabels):

        f1Score1=metrics.adjusted_rand_score(trueLabels,self.labels)
        print(f1Score1)


def run(fpath, pdir):

    pipe=pipeLine()
    pipe.readData(fpath,pdir)
    pipe.skLearnKmeans()
    #print(pipe.fileDictionary)
    for i in range(len(pipe.labels)):
        print(pipe.fileNames[i],":",pipe.labels[i])
    jsonLoad=open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/imdb/movies.json'),"r")
    labelJson=json.load(jsonLoad)
    trueLabels=[]
    #for fName in pipe.fileNames:
        #trueLabels.append(labelJson[fName])

    print(pipe.labels," len",len(pipe.labels))
    print("True labels>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    trueLabels1=np.asarray([0,1,2,2,0,1,3,4,2,1])

    print(trueLabels1,"labels",len(trueLabels1))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Cluster count",pipe.clusterCount)
    #pipe.calculateF1Score(trueLabels1)
    pipe.getNamedEntities()
    pipe.sendToPlotData()
    #im.plotClusters(normalized,fileNames,labels,centroids,True)



def runForAPI(fpath,pdir):
    pipe = pipeLine()
    pipe.readData(fpath, pdir)
    pipe.skLearnKmeans()
    pipe.getNamedEntitiesAPI()
    #print(pipe.entities)
    finalData=list()
    for each in pipe.fileDictionary:
        temp=dict()
        temp['files']=pipe.fileDictionary[each]
        temp['summary']=pipe.entities['summary'][each]
        finalData.append(temp)


    return {"dict":finalData}