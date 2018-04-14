from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import style
style.use("ggplot")
import re
import nltk
from gensim.models.keyedvectors import KeyedVectors

removableWords = set(stopwords.words('english'))

punctuations = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—']

removableWords.update(punctuations)
vectorSize = 300

temp = open("/home/sharathbhragav/PycharmProjects/nlp_pipeline/models/harryPotterFullWord2VecModelSize300", "rb")
trainingModelGoogle = pickle.load(temp)
modelUsed = trainingModelGoogle


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
    cleaned = [word.lower() for word in temp1 if word not in removableWords and len(word)>0]
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
    wordVector1 = np.array(modelUsed[inputWord])  # trainingModelGoogle
    return wordVector1

def getWordVectorAbsoluteValue(wordVector):
    sum =0
    for i in wordVector:
        sum = sum+ (i*i)
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


def getSentancesListFromDoc(documentHandle,stopWordsRequired):
    sentances = splitCorpusIntoSentances(documentHandle)
    docWords= []
    for sent in sentances:
        if stopWordsRequired:
            words = tokanizeSingleSentance(sent)
        else:
            words= tokanizeAndRemoveStopWordsSingleSentance(sent)
        if len(words)>0:
            docWords.append(words)
    return docWords


def getDocVector(documentHandle,stopWordsRequired=False):
    totalDocVec = np.array([float(0.0) for x in range(vectorSize)])
    countOfWords = 0
    countOfIgnore=0
    completeList = getSentancesListFromDoc(documentHandle,stopWordsRequired)
    ignoredWords=open('documents/ignoredWords','w')
    #print(completeList)
    for sentances in completeList:
        #print(sentances)
        for word in sentances:
            try:
                wordVec = getWordVector(word)
                #print("wordvec>>>>>>>>>>>>>>>",wordVec)
                countOfWords = countOfWords + 1
                #print("count>>>>>>>>>>>>>>>>>>>>>",countOfWords)
                totalDocVec += wordVec
                #print(word, ">>>>>>>>", totalDocVec)
            except:
                countOfIgnore+=1

                continue
    totalDocVec /= countOfWords
    ignoredWords.write("Ignored count="+str(countOfIgnore)+"\n")
    ignoredWords.write("counted="+str(countOfWords))
    ignoredWords.close()
    return totalDocVec

def getIgnoreWordsPercentage():
    ignored=open('documents/ignoredWords','r')
    readWords1=ignored.readline()
    readWord2=ignored.readline()
    ignoredCount=(int)(readWords1[15:])
    counted=(int)(readWord2[8:])
    percent=ignoredCount/(ignoredCount+counted)
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
    similarity = modelUsed.similarity(word1, word2)
    return similarity


def plotDocumentWords(documentHandle123,stopWordsRequired=False):
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
    return (totalDocVec,correspondingWord)

def getCommonWordsBetweenDocs(documentHandle1,documentHandle2):
    set1=[]
    set2=[]
    completeList = getSentancesListFromDoc(documentHandle1,False)
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

    s1=set(set1)
    s2=set(set2)
    return s1.intersection(s2)


def getPlotValuesOfDocuments(documentHandles):
    vectors = []
    for handle in documentHandles:
        vec=getDocVector(handle)
        if(len(vec)>0):
            vectors.append(vec)

    docArray=np.asarray(vectors,dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(docArray)
    return pcaOut


def compressWordVecToPlot(wordVecList):
    numArray = np.asarray(wordVecList,dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(numArray)
    return pcaOut


def plotDocument(documentHandle,StopWordsRequired=False):
    (wordVecList,wordList) = im.plotDocumentWords(documentHandle,StopWordsRequired)
    plotData = compressWordVecToPlot(wordVecList)
    x=[]
    y=[]
    for k in plotData:
        x.append(k[0])
        y.append(k[1])
    plt.scatter(x,y,linewidths=2,s=5)
    for i in range(len(wordList)):
        xy=(x[i],y[i])
        plt.annotate(wordList[i],xy)
    plt.show()
