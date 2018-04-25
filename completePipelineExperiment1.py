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
#pathToData="/media/sharathbhragav/New Volume/redditPosts/hot/"
#pathToPickles="/media/sharathbhragav/New Volume/redditPosts/pickles/"

pathToData="datasets/custom2/"
pathToPickles = "datasets/custom2/"


removableWords = set(stopwords.words('english'))

extraWords = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—', ' ','Reddit','reddit','Lol','Nah','I']

removableWords.update(extraWords)
def getNamedEntties(path,fileDictionary,numberOfEntities=5):
    organizations = {}
    persons = {}
    places = {}
    locations={}
    nouns={}
    completeSummery={}
    for i in range(len(fileDictionary)):
        clusterOrganization = []
        clusterPerson = []
        clusterPlace = []
        clusterLocation = []
        clusterNouns = []
        clusterSummery=[]
        for docName in fileDictionary[i]:
            docTemp = open(path + docName, "r")
            docTempRead=docTemp.read().replace("\n",' ')
            doc = nlp(docTempRead)
            for eachNoun in doc.noun_chunks:
                if str(eachNoun).lower() not in removableWords:
                    clusterNouns.append(str(eachNoun))

            for ent in doc.ents:
                if ent.text not in removableWords:
                    # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    if ent.label_ == "ORG":
                        clusterOrganization.append(ent.text)
                        clusterSummery.append(ent.text)
                    if ent.label_ == 'PERSON':
                        clusterPerson.append(ent.text)
                        clusterSummery.append(ent.text)

                    if ent.label_ == "GPE":
                        clusterPlace.append(ent.text)
                        clusterSummery.append(ent.text)
                    if ent.label_ == "LOC":
                        clusterLocation.append(ent.text)
                        clusterSummery.append(ent.text)
            #summer_freq = Counter(clusterSummery)
            #clusterSummery=summer_freq.most_common(10)
            clusterSummery.append("\n")
            docTemp.close()
        #clusterNouns = list(set(clusterNouns))
        organizations[i] = clusterOrganization
        persons[i] = clusterPerson
        places[i] = clusterPlace
        locations[i]=clusterLocation
        nouns[i]=clusterNouns
        completeSummery[i]=clusterSummery
        organizations_freq = Counter(clusterOrganization)
        persons_freq = Counter(clusterPerson)
        places_freq = Counter(clusterPlace)
        locations_freq=Counter(clusterLocation)
        noun_freq = Counter(clusterNouns)
        completeSummery_freq = Counter(clusterSummery)
        print("Cluster:", i)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Organizations")
        print(organizations_freq.most_common(numberOfEntities))
        print("Persons")
        print(persons_freq.most_common(numberOfEntities))
        print("Places")
        print(places_freq.most_common(numberOfEntities))
        print("Locations")
        print(locations_freq.most_common(numberOfEntities))
        print("Noun list")
        print(noun_freq.most_common(20))

        print("Complete summery"," $ ".join(set(clusterSummery)))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")










custom2Names=open(pathToPickles+"plotNamesOfDocs","rb")
fileNames=pickle.load(custom2Names)
custom2Pickle=open(pathToPickles+"plotValuesOfDocs","rb")
total1=pickle.load(custom2Pickle)
print(fileNames)

normalized=normalize(total1)


(clusterCount,clf)=exp2.customKMeansComplete(normalized,fileNames)

labels=clf.getLabels(normalized)

fileNameDictionary=exp2.getDocClustersNames(clusterCount,labels,fileNames)

print(fileNameDictionary)

getNamedEntties(pathToData,fileNameDictionary,10)

im.plotClusters(normalized,fileNames,labels,clf.centroids,True)
