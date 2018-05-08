from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from gensim.models.keyedvectors import KeyedVectors

from sklearn.decomposition import PCA
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from operator import itemgetter
from random import randint
from enum import Enum
from sklearn.cluster import Birch
import spacy
from . import kMeans
from collections import Counter
from django.conf import settings

removableWords = set(stopwords.words('english'))

extraWords = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
              '&', '*', '(', ')', '-', '_', '=', '+', '—', ' ', 'Reddit', 'reddit', 'Lol', 'Nah', 'I']

removableWords.update(extraWords)
vectorSize = 300

trainingModelGoogle = KeyedVectors.load_word2vec_format(os.path.join(settings.BASE_DIR, 'models/GoogleNews-vectors-negative300.bin'), binary=True, limit=10000)
nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')




def printStopWords():
    print(removableWords)


def setModel(inputModel):
    global modelUsed
    modelUsed = inputModel


def splitCorpusIntoSentances(fileHandle):
    corpus = fileHandle.read()
    sentances = sent_tokenize(corpus)
    return sentances


def tokanizeAndRemoveStopWordsSingleSentance(inputSentance):
    temp1 = word_tokenize(inputSentance)
    cleaned = [word.lower() for word in temp1 if word not in removableWords and len(word) > 0]
    return cleaned


def tokanizeSingleSentance(inputSentance):
    temp1 = word_tokenize(inputSentance)
    return temp1


def getNormalizedVector(inputVector):
    sqSum = 0
    for k in inputVector:
        sqSum = sqSum + (k * k)
    normalizedVector = inputVector / math.sqrt(sqSum)
    return normalizedVector


def getWordVector(inputWord):
    trainingModelGoogle = KeyedVectors.load_word2vec_format(
        os.path.join(settings.BASE_DIR, 'models/GoogleNews-vectors-negative300.bin'), binary=True, limit=10000)
    modelUsed = trainingModelGoogle
    wordVector1 = np.array(modelUsed[inputWord])  # trainingModelGoogle
    return wordVector1


def getWordVectorAbsoluteValue(wordVector):
    sum = 0
    for i in wordVector:
        sum = sum + (i * i)
    return math.sqrt(sum)


def getSentanceVector(inputSentance):
    cleanedSentance = tokanizeAndRemoveStopWordsSingleSentance(inputSentance)
    sentanceVector = np.array([float(0.0) for x in range(vectorSize)])
    for w in cleanedSentance:
        tempWord = getWordVector(w)
        sentanceVector += tempWord
    lengthOfSentence = len(cleanedSentance)
    sentanceVector /= lengthOfSentence
    return sentanceVector


def getSentancesListFromDoc(documentHandle, stopWordsRequired):
    sentances = splitCorpusIntoSentances(documentHandle)
    docWords = []
    for sent in sentances:
        if stopWordsRequired:
            words = tokanizeSingleSentance(sent)
        else:
            words = tokanizeAndRemoveStopWordsSingleSentance(sent)
        if len(words) > 0:
            docWords.append(words)
    return docWords


def getDocVector(documentHandle, stopWordsRequired=False):
    totalDocVec = np.array([float(0.0) for x in range(vectorSize)])
    countOfWords = 1
    countOfIgnore = 0
    completeList = getSentancesListFromDoc(documentHandle, stopWordsRequired)
    #ignoredWords = open('/home/sharathsbhragav/PycharmProjects/nlp_pipeline/documents/ignoredWords', 'w')
    # print(completeList)
    for sentances in completeList:
        # print(sentances)
        for word in sentances:
            try:
                wordVec = getWordVector(word)
                # print("wordvec>>>>>>>>>>>>>>>",wordVec)
                countOfWords += 1
                # print("count>>>>>>>>>>>>>>>>>>>>>",countOfWords)
                totalDocVec += wordVec
                # print(word, ">>>>>>>>", totalDocVec)
            except:
                countOfIgnore += 1
                continue
    totalDocVec /= countOfWords
    #ignoredWords.write("Ignored count=" + str(countOfIgnore) + "\n")
    #ignoredWords.write("counted=" + str(countOfWords))
    #ignoredWords.close()
    return totalDocVec


def getIgnoreWordsPercentage():
    ignored = open('documents/ignoredWords', 'r')
    readWords1 = ignored.readline()
    readWord2 = ignored.readline()
    ignoredCount = (int)(readWords1[15:])
    counted = (int)(readWord2[8:])
    percent = ignoredCount / (ignoredCount + counted)
    return percent


def getSentanceSimilarity(sentance1, sentance2):
    sentenceVector1 = getSentanceVector(sentance1)
    sentenceVector2 = getSentanceVector(sentance2)
    sentenceVector1 = getNormalizedVector(sentenceVector1)
    sentenceVector2 = getNormalizedVector(sentenceVector2)
    similarity = np.dot(sentenceVector1, sentenceVector2)
    return similarity


def getDocSimilarity(docVector1, docVector2):
    normalizedDocVec1 = getNormalizedVector(docVector1)
    normalizedDocVec2 = getNormalizedVector(docVector2)
    similarity = np.dot(normalizedDocVec1, normalizedDocVec2)
    return similarity


def getWordSimilarity(word1, word2):
    wordVec1 = getWordVector(word1)
    wordVec2 = getWordVector(word2)
    normalizedWord1 = getNormalizedVector(wordVec1)
    normalizedWord2 = getNormalizedVector(wordVec2)
    similarity = np.dot(normalizedWord1, normalizedWord2)
    return similarity


def getWord2VecWordSimilarity(word1, word2):

    modelUsed = trainingModelGoogle
    similarity = modelUsed.similarity(word1, word2)
    return similarity


def plotDocumentWords(documentHandle123, stopWordsRequired=False):
    totalDocVec = []
    correspondingWord = []
    countOfWords = 0
    completeList = getSentancesListFromDoc(documentHandle123, stopWordsRequired)
    for sentances in completeList:
        for word in sentances:
            try:
                wordVec = getWordVector(word)

                totalDocVec.append(wordVec)
                correspondingWord.append(word)
            except:
                continue
    return (totalDocVec, correspondingWord)


def getCommonWordsBetweenDocs(documentHandle1, documentHandle2):
    set1 = []
    set2 = []
    completeList = getSentancesListFromDoc(documentHandle1, False)
    # print(completeList)
    for sentances in completeList:
        # print(sentances)
        for word in sentances:
            set1.append(word)

    completeList = getSentancesListFromDoc(documentHandle2, False)
    # print(completeList)
    for sentances in completeList:
        # print(sentances)
        for word in sentances:
            set2.append(word)

    s1 = set(set1)
    s2 = set(set2)
    return s1.intersection(s2)


def getPlotValuesOfDocuments(documentHandles):
    vectors = []
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(documentHandles)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for handle in documentHandles:
        try:
            vec = getDocVector(handle)
            if (len(vec) > 0):
                vectors.append(vec)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            #print(handle, " failed to read")
    docArray = np.asarray(vectors, dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(docArray)
    return pcaOut


def compressWordVecToPlot(wordVecList):
    numArray = np.asarray(wordVecList, dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(numArray)
    return pcaOut


def plotDocument(documentHandle, StopWordsRequired=False):
    (wordVecList, wordList) = plotDocumentWords(documentHandle, StopWordsRequired)
    plotData = compressWordVecToPlot(wordVecList)
    x = []
    y = []
    for k in plotData:
        x.append(k[0])
        y.append(k[1])
    plt.scatter(x, y, linewidths=2, s=5)
    for i in range(len(wordList)):
        xy = (x[i], y[i])
        plt.annotate(wordList[i], xy)
    plt.show()


def plotClusters(eachPointList, eachFileNameList, labels, centroids, anotate=False):
    colors = 100 * ["r", "g", "b", "c", "k", "y", "m", "#DD2C00", "#795548", "#1B5E20", "#0091EA", "#6200EA", "#311B92",
                    "#880E4F"]

    count = 0

    for centroid in centroids:

        plt.scatter(centroid[0], centroid[1], marker="o", color=colors[count], s=75,
                    linewidths=5)
        if anotate:
            xy = (centroid[0], centroid[1])
            plt.annotate("Cluster " + str(count), xy)
        count = count + 1
    point = 0
    for label in labels:
        plt.scatter(eachPointList[point][0], eachPointList[point][1], marker='x', s=30, color=colors[label],
                    linewidths=5)
        point += 1

    print("point==", point)
    print("Count=", count)
    plt.show()


class ClusteringAlgorithm(Enum):
    skLearnKMeans = 1
    customKMeans = 2
    skLearnBirch = 3


def getOptimalClustersSilhoutte(data, algorithm=ClusteringAlgorithm.skLearnKMeans):
    silhoutteScores = {}
    rotationStored = {}
    thresholdValues = {}
    if algorithm == ClusteringAlgorithm.customKMeans:
        for clusterKmeansNumber in range(2, 20):
            try:
                clf = kMeans.K_Means(clusterKmeansNumber, tolerance=0.00001, max_iterations=800)
                rotation = randamozieSeed(data, clusterKmeansNumber)
                clf.fit(data, spherical=True, rotationArray=rotation)
                labels = clf.getLabels(data)
                silhouette_avg = silhouette_score(data, labels)
                silhoutteScores[clusterKmeansNumber] = silhouette_avg
                rotationStored[clusterKmeansNumber] = rotation
                # print(clusterKmeansNumber,">>>>>>>",rotation)
            except:
                continue
                # print(clusterKmeansNumber," chucked")
    elif algorithm == ClusteringAlgorithm.skLearnKMeans:
        for clusterKmeansNumber in range(2, 20):
            clf = KMeans(n_clusters=clusterKmeansNumber)
            labels = clf.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhoutteScores[clusterKmeansNumber] = silhouette_avg

    elif algorithm == ClusteringAlgorithm.skLearnBirch:
        for i in range(2, 100):
            brc = Birch(branching_factor=50, n_clusters=None, threshold=0.01 * i, compute_labels=True)

            labels = brc.fit_predict(data)
            print(len(labels))
            try:
                silhouette_avg = silhouette_score(data, labels)
                clusterNumber = len(set(labels))
                silhoutteScores[clusterNumber] = silhouette_avg
                thresholdValues[clusterNumber] = i * 0.01

            except:
                continue

    sortedSil = sorted(silhoutteScores.items(), key=itemgetter(1))
    selectedClusterNumber = sortedSil[-1][0]
    print("selected number of clusters=", selectedClusterNumber)
    if algorithm == ClusteringAlgorithm.customKMeans:
        return (selectedClusterNumber, rotationStored[selectedClusterNumber])
    elif algorithm == ClusteringAlgorithm.skLearnBirch:
        return selectedClusterNumber
    else:
        return thresholdValues[selectedClusterNumber]


def skLearnKMeansComplete(data):
    selectedClusterNumber = getOptimalClustersSilhoutte(data, ClusteringAlgorithm.skLearnKMeans)
    clf = KMeans(n_clusters=selectedClusterNumber)
    clf.fit(data)
    # centroidsKmeans=clf.cluster_centers_
    # labelsKmeans=clf.labels_

    return (selectedClusterNumber, clf)


def randamozieSeed(data, k):
    outputSeed = []
    for randomNumberIter in range(k):
        random = randint(0, len(data))
        outputSeed.append(data[random])
    return outputSeed


def customKMeansComplete(data):
    (selectedClusterNumber, rotation) = getOptimalClustersSilhoutte(data, ClusteringAlgorithm.customKMeans)

    clf = kMeans.K_Means(selectedClusterNumber, tolerance=0.00001, max_iterations=800)

    clf.fit(data, spherical=True, rotationArray=rotation)
    # classifications=clf.classifications
    # centroids=clf.centroids

    return (selectedClusterNumber, clf)


def skLearnBirch(data):
    threshold = getOptimalClustersSilhoutte(data, ClusteringAlgorithm.skLearnBirch)
    brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
    labels = brc.fit_predict(data)
    selectedClusterNumber = len(brc.subcluster_centers_)

    return (selectedClusterNumber, brc)


def getDocClustersNames(clusterCount, labels, fileNames):
    fileNamesClusters = {}
    labelsOfDocs = labels
    for clusterNumber in range(clusterCount):
        singleClusterDocs = []
        for getDocData in range(len(labels)):
            if labelsOfDocs[getDocData] == clusterNumber:
                singleClusterDocs.append(fileNames[getDocData])
        fileNamesClusters[clusterNumber] = singleClusterDocs

    return fileNamesClusters


def getNamedEntties(path, fileDictionary, numberOfEntities=5, summaryLimitWords=25):

    organizations = {}
    persons = {}
    places = {}
    locations = {}
    nouns = {}
    completeSummery = {}
    for i in range(len(fileDictionary)):
        clusterOrganization = []
        clusterPerson = []
        clusterPlace = []
        clusterLocation = []
        clusterNouns = []
        clusterSummery = []
        for docName in fileDictionary[i]:
            docTemp = open(os.path.join(path, docName), "r")
            docTempRead = docTemp.read().replace("\n", ' ')
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
            # summer_freq = Counter(clusterSummery)
            # clusterSummery=summer_freq.most_common(10)
            clusterSummery.append("\n")
            docTemp.close()
        # clusterNouns = list(set(clusterNouns))
        organizations[i] = clusterOrganization
        persons[i] = clusterPerson
        places[i] = clusterPlace
        locations[i] = clusterLocation
        nouns[i] = clusterNouns
        completeSummery[i] = clusterSummery

        organizations_freq = Counter(clusterOrganization)
        persons_freq = Counter(clusterPerson)
        places_freq = Counter(clusterPlace)
        locations_freq = Counter(clusterLocation)
        noun_freq = Counter(clusterNouns)

        '''
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
        print(noun_freq.most_common(summaryLimitWords))
        for each in noun_freq.most_common(20):
            clusterSummery.append(each[0])
        completeSummery_freq = Counter(clusterSummery)
        print("Limited summery", completeSummery_freq.most_common(summaryLimitWords))
        print("\n\n\n")
        print("Complete summery", " $ ".join(set(clusterSummery)))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        '''
    entities = dict()
    entities['org'] = organizations
    entities['persons'] = persons
    entities['places'] = places
    entities['loc'] = locations
    entities['nouns'] = nouns
    entities['summary'] = completeSummery
    return entities
