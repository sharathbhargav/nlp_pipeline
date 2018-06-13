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
from .algorithms import kMeans
from collections import Counter
from django.conf import settings
import pdb
from nltk.tokenize import RegexpTokenizer
from spacy import displacy



tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
removableWords = set(stopwords.words('english'))
extraWords = [ '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%',
                        '^', '&', '*', '(', ')', '-', '_', '=', '+', '—',r' *']
removableWords.update(extraWords)
vectorSize = 300

#print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#print(settings.PROJECT_ROOT)
#print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
trainingModelGoogle = KeyedVectors.load_word2vec_format(os.path.join(settings.BASE_DIR, 'nlpPipeline1/backend/models/GoogleNews-vectors-negative300.bin'), binary=True, limit=10000)
nlp = spacy.load('/home/sharathbhragav/anaconda3/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

modelUsed=trainingModelGoogle


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
    tokenized_line = tokenizer.tokenize(inputSentance)
    cleaned = [word for word in tokenized_line if word not in removableWords]
    """
    temp1 = word_tokenize(inputSentance)
    cleaned = [word.lower() for word in temp1 if word not in removableWords and len(word) > 0]
    """
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
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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
    hybrid = 4


def getOptimalClustersSilhoutte(data, algorithm=ClusteringAlgorithm.skLearnKMeans):
    silhoutteScores = {}
    rotationStored = {}
    thresholdValues = {}
    hybridValues={}
    kmeansClusterNumberRange= range(2,len(data)//2)
    if algorithm == ClusteringAlgorithm.customKMeans:
        for clusterKmeansNumber in kmeansClusterNumberRange:
            try:
                #pdb.set_trace()

                try:
                    clf = kMeans.K_Means(clusterKmeansNumber, tolerance=0.0001, max_iterations=800)
                except:
                    print("k class failed")


                try:
                    rotation = randamozieSeed(data, clusterKmeansNumber)
                    rotationStored[clusterKmeansNumber] = rotation
                    clf.fit(data, spherical=True, rotationArray=rotation)
                    labels = clf.getLabels(data)
                    silhouette_avg = silhouette_score(data, labels)
                    silhoutteScores[clusterKmeansNumber] = silhouette_avg


                except:
                    print("fit failed, label failed,k==",clusterKmeansNumber)

                    print("sil score failed")
                    print("labels",len(labels))
                    print("data",len(data))


                # print(clusterKmeansNumber,">>>>>>>",rotation)
            except:
                print(clusterKmeansNumber, " chucked custom kmeans")
                continue

    elif algorithm == ClusteringAlgorithm.skLearnKMeans:
        for clusterKmeansNumber in kmeansClusterNumberRange:
            clf = KMeans(n_clusters=clusterKmeansNumber)
            labels = clf.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhoutteScores[clusterKmeansNumber] = silhouette_avg

    elif algorithm == ClusteringAlgorithm.skLearnBirch:
        for i in range(2, 100):
            brc = Birch(branching_factor=50, n_clusters=None, threshold=0.01 * i, compute_labels=True)

            labels = brc.fit_predict(data)
            #print(len(labels))
            try:
                silhouette_avg = silhouette_score(data, labels)
                clusterNumber = len(set(labels))
                silhoutteScores[clusterNumber] = silhouette_avg
                thresholdValues[clusterNumber] = i * 0.01

            except:
                continue
    elif algorithm == ClusteringAlgorithm.hybrid:

        for clusterKmeansNumber in kmeansClusterNumberRange:
            try:
                #pdb.set_trace()

                try:
                    clf = kMeans.K_Means(clusterKmeansNumber, tolerance=0.0001, max_iterations=800)
                except:
                    print("k class failed")


                try:
                    rotation = randamozieSeed(data, clusterKmeansNumber)
                    rotationStored[clusterKmeansNumber] = rotation
                    clf.fit(data, spherical=True, rotationArray=rotation)
                    labels = clf.getLabels(data)
                    silhouette_avg = silhouette_score(data, labels)
                    silhoutteScores[clusterKmeansNumber] = silhouette_avg
                    hybridValues[clusterKmeansNumber]='c'


                except:
                    print("fit failed, label failed,k==",clusterKmeansNumber)

                    print("sil score failed")
                    print("labels",len(labels))
                    print("data",len(data))


                # print(clusterKmeansNumber,">>>>>>>",rotation)
            except:
                print(clusterKmeansNumber, " chucked custom kmeans")
                continue


        for clusterKmeansNumber in kmeansClusterNumberRange:
            clf = KMeans(n_clusters=clusterKmeansNumber)
            labels = clf.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhoutteScores[clusterKmeansNumber] = silhouette_avg
            hybridValues[clusterKmeansNumber]='s'








    #print("Rotation array>>>>>>>>",rotationStored)
    #print("Sill arr",silhoutteScores)
    sortedSil = sorted(silhoutteScores.items(), key=itemgetter(1))
    #print("In sorted sil>>>>>>>>>>>>>>>>>>>>>>>>>",sortedSil)
    selectedClusterNumber = sortedSil[-1][0]

    if algorithm ==ClusteringAlgorithm.hybrid:
        if hybridValues[selectedClusterNumber]=='c':
            clf = kMeans.K_Means(selectedClusterNumber, tolerance=0.0001, max_iterations=800)
            clf.fit(data, spherical=True, rotationArray=rotationStored[selectedClusterNumber])
            labels = clf.getLabels(data)
            silhouette_avg = silhouette_score(data, labels)
            return (ClusteringAlgorithm.customKMeans, selectedClusterNumber, rotationStored[selectedClusterNumber])
        else:
            clf = KMeans(n_clusters=selectedClusterNumber)
            labels = clf.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            return (ClusteringAlgorithm.skLearnKMeans, selectedClusterNumber)



    print("Clustered selected", selectedClusterNumber)
    print("selected number of clusters=", selectedClusterNumber)
    if algorithm == ClusteringAlgorithm.customKMeans:
        return (selectedClusterNumber, rotationStored[selectedClusterNumber])
    elif algorithm == ClusteringAlgorithm.skLearnKMeans:
        return selectedClusterNumber
    else :
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

    clf = kMeans.K_Means(selectedClusterNumber, tolerance=0.0001, max_iterations=800)

    clf.fit(data, spherical=True, rotationArray=rotation)
    # classifications=clf.classifications
    # centroids=clf.centroids

    return (selectedClusterNumber, clf)


def hybridKmeans(data):
    outTuple = getOptimalClustersSilhoutte(data, ClusteringAlgorithm.hybrid)
    if len(outTuple)>2:
        clf = kMeans.K_Means(outTuple[1], tolerance=0.0001, max_iterations=800)

        clf.fit(data, spherical=True, rotationArray=outTuple[2])
        # classifications=clf.classifications
        # centroids=clf.centroids
        print("Custom k means >>>",outTuple[1])
        return (0,outTuple[1], clf)
    else:
        clf = KMeans(n_clusters=outTuple[1])
        clf.fit(data)
        # centroidsKmeans=clf.cluster_centers_
        # labelsKmeans=clf.labels_
        print("sk learn k means",outTuple[1])
        return (1,outTuple[1], clf)


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
    fileOrgs={}
    filePersons={}
    filePlaces={}
    fileLocations={}
    fileNouns={}
    fileSummary={}
    completeSummery = {}
    for i in range(len(fileDictionary)):
        clusterOrganization = []
        clusterPerson = []
        clusterPlace = []
        clusterLocation = []
        clusterNouns = []
        clusterSummery = []

        for docName in fileDictionary[i]:
            eachOrg=[]
            eachPerson=[]
            eachPlace=[]
            eachLocation=[]
            eachNoun=[]
            eachSummery=[]
            docTemp = open(os.path.join(path, docName), "r")
            """
            sentList=splitCorpusIntoSentances(docTemp)
            cleanedList=[]
            for eachSent in sentList:
                cleaned=tokanizeAndRemoveStopWordsSingleSentance(eachSent)
                cleanedList.append(' '.join(cleaned))
            docTemp.seek(0)
            """
            doc = nlp(docTemp.read().replace('\n',' '))
            for eachNoun1 in doc.noun_chunks:
                if str(eachNoun1).lower() not in removableWords:
                    clusterNouns.append(str(eachNoun1))
                    eachNoun.append(str(eachNoun1))

            for ent in doc.ents:
                if ent.text not in removableWords:
                    # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    if ent.label_ == "ORG":
                        clusterOrganization.append(ent.text)
                        eachOrg.append(ent.text)
                        clusterSummery.append(ent.text)
                        eachSummery.append(ent.text)
                    if ent.label_ == 'PERSON':
                        eachPerson.append(ent.text)
                        clusterPerson.append(ent.text)
                        clusterSummery.append(ent.text)
                        eachSummery.append(ent.text)
                    if ent.label_ == "GPE":
                        eachPlace.append(ent.text)
                        clusterPlace.append(ent.text)
                        clusterSummery.append(ent.text)
                        eachSummery.append(ent.text)
                    if ent.label_ == "LOC":
                        eachLocation.append(ent.text)
                        clusterLocation.append(ent.text)
                        clusterSummery.append(ent.text)
                        eachSummery.append(ent.text)
            # summer_freq = Counter(clusterSummery)
            # clusterSummery=summer_freq.most_common(10)
            organizations_freq = Counter(eachOrg)
            persons_freq = Counter(eachPerson)
            places_freq = Counter(eachPlace)
            locations_freq = Counter(eachLocation)
            noun_freq = Counter(eachNoun)
            summary_freq=Counter(eachSummery)
            orgCount=10#int(0.20*(len(eachOrg)))
            personCount=10#int(0.20*(len(eachPerson)))
            placesCount=10#int(0.20*(len(eachPlace)))
            locCount=10#int(0.20*(len(eachLocation)))
            nounCount=25#int(0.20*(len(eachNoun)))

            fileOrgs[docName]=[ent[0] for ent in organizations_freq.most_common(orgCount)]
            filePersons[docName]=[ent[0] for ent in persons_freq.most_common(personCount)]
            filePlaces[docName]=[ent[0] for ent in places_freq.most_common(placesCount)]
            fileLocations[docName]=[ent[0] for ent in locations_freq.most_common(locCount)]
            fileNouns[docName]=[ent[0] for ent in noun_freq.most_common(nounCount)]
            fileSummary[docName] = [ent[0] for ent in summary_freq.most_common(25)]
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #print(nounCount)
            #print(fileNouns)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            clusterSummery.append("\n")
            docTemp.close()
        # clusterNouns = list(set(clusterNouns))
        organizations[i] = [ent[0] for ent in Counter(clusterOrganization).most_common(10)]
        persons[i] = [ent[0] for ent in Counter(clusterPerson).most_common(10)]
        places[i] = [ent[0] for ent in Counter(clusterPlace).most_common(10)]
        locations[i] = [ent[0] for ent in Counter(clusterLocation).most_common(10)]
        nouns[i] = [ent[0] for ent in Counter(clusterNouns).most_common(10)]
        completeSummery[i] = [ent[0] for ent in Counter(clusterSummery).most_common(10)]



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
    entities['file_org'] = fileOrgs
    entities['file_persons'] = filePersons
    entities['file_places'] = filePlaces
    entities['file_loc'] = fileLocations
    entities['file_nouns'] = fileNouns
    entities['file_summary']=fileSummary

    return entities

def getNamedEntitiesForAPI(path,fileDictionary):
    totalSummary = dict()
    for i in range(len(fileDictionary)):
        eachSummary = []
        for docName in fileDictionary[i]:

            docTemp = open(os.path.join(path, docName), "r")
            docTempRead = docTemp.read().replace("\n", ' ')
            doc = nlp(docTempRead)
            for ent in doc.ents:
                if ent.text not in removableWords:
                    if ent.label_ == "ORG":
                        eachSummary.append(ent.text)
                    if ent.label_ == "PERSON":
                        eachSummary.append(ent.text)
                    if ent.label_ == "GPE":
                        eachSummary.append(ent.text)
                    if ent.label_ == "LOC":
                        eachSummary.append(ent.text)

        totalSummary[i]=[ent[0] for ent in Counter(eachSummary).most_common(20)]
    entities = dict()
    entities['summary']=totalSummary
    return entities


def getNamedEntitiesForSingleFile(fileHandle):
    data=fileHandle.read()
    doc = nlp(data.replace('\n', ' '))
    eachSummary=[]
    for eachNoun1 in doc.noun_chunks:
        if str(eachNoun1).lower() not in removableWords:

            eachSummary.append(str(eachNoun1))
    for ent in doc.ents:
        if ent.text not in removableWords:
            if ent.label_ == "ORG":
                eachSummary.append(ent.text)
            if ent.label_ == "PERSON":
                eachSummary.append(ent.text)
            if ent.label_ == "GPE":
                eachSummary.append(ent.text)
            if ent.label_ == "LOC":
                eachSummary.append(ent.text)

    summary=  [ent[0] for ent in Counter(eachSummary).most_common(20)]

    return summary