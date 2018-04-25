from gensim.models.keyedvectors import KeyedVectors
import individualModules as im
import numpy as np
from nltk.corpus import stopwords
import math
import pickle
from sklearn.preprocessing import normalize
from collections import Counter
import clusteringExperiments2 as exp2
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from matplotlib import style

trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)
nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')
pathToData="/media/sharathbhragav/New Volume/redditPosts/hot/"
pathToPickles="/media/sharathbhragav/New Volume/redditPosts/pickles/"

colors = 100 * ["r", "g", "b", "c", "k"]
removableWords = set(stopwords.words('english'))

extraWords = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—', ' ','Reddit','reddit','Lol','Nah']

removableWords.update(extraWords)
def getNamedEntties(path,fileDictionary,numberOfEntities=5):
    organizations = {}
    persons = {}
    places = {}
    for i in range(len(fileDictionary)):
        clusterOrganization = []
        clusterPerson = []
        clusterPlace = []

        for docName in fileDictionary[i]:
            docTemp = open(path + docName, "r")
            docTempRead=docTemp.read().replace("\n",' ')
            doc = nlp(docTempRead)
            for ent in doc.ents:
                if ent.text not in removableWords:
                    # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    if ent.label_ == "ORG":
                        clusterOrganization.append(ent.text)
                    if ent.label_ == 'PERSON':
                        clusterPerson.append(ent.text)
                    if ent.label_ == "GPE":
                        clusterPlace.append(ent.text)
            docTemp.close()
        organizations[i] = clusterOrganization
        persons[i] = clusterPerson
        places[i] = clusterPlace
        organizations_freq = Counter(clusterOrganization)
        persons_freq = Counter(clusterPerson)
        places_freq = Counter(clusterPlace)
        print("Cluster:", i)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Organizations")
        print(organizations_freq.most_common(numberOfEntities))
        print("Persons")
        print(persons_freq.most_common(numberOfEntities))
        print("Places")
        print(places_freq.most_common(numberOfEntities))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")





def plotClusters(eachPointList, eachFileNameList, labels, centroids, anotate=False):
    count = 0
    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], marker="o", color=colors[count], s=75,
                    linewidths=5)
        if anotate:
            xy=(centroids[centroid][0],centroids[centroid][1])
            plt.annotate("Cluster "+str(count), xy)
        count = count + 1
    point = 0
    for label in labels:
        plt.scatter(eachPointList[point][0], eachPointList[point][1],marker='x', s=30, color=colors[label],linewidths=5)
        point+=1

    print("point==",point)
    print("Count=",count)
    plt.show()




custom2Names=open(pathToPickles+"plotNamesOfDocs","rb")
fileNames=pickle.load(custom2Names)
custom2Pickle=open(pathToPickles+"plotValuesOfDocs","rb")
total1=pickle.load(custom2Pickle)
print(fileNames)

normalized=normalize(total1)
colors = 100 * ["r", "g", "b", "c", "k","y","m","#DD2C00","#795548","#1B5E20","#0091EA","#6200EA","#311B92","#880E4F"]

(clusterCount,clf)=exp2.customKMeansComplete(normalized,fileNames)

labels=clf.getLabels(normalized)

fileNameDictionary=exp2.getDocClustersNames(clusterCount,labels,fileNames)

print(fileNameDictionary)

#getNamedEntties(pathToData,fileNameDictionary,10)

plotClusters(normalized,fileNames,labels,clf.centroids,True)
