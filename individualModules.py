from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
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

#temp = open("/home/sharathbhragav/PycharmProjects/nlp_pipeline/models/harryPotterFullWord2VecModelSize300", "rb")
#trainingModelGoogle = pickle.load(temp)
modelUsed = True


def setModel(inputModel):
    global modelUsed
    modelUsed = inputModel


def splitCorpusIntoSentances(fileHandle):
    corpus = fileHandle.read()
    sentances = sent_tokenize(corpus)
    return sentances


def tokanizeAndRemoveStopWordsSingleSentance(inputSentance):
    temp1 = word_tokenize(inputSentance)
    cleaned = [word for word in temp1 if word not in removableWords and len(word)>0]
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
    completeList = getSentancesListFromDoc(documentHandle,stopWordsRequired)
    for sentances in completeList:
        for word in sentances:
            try:
                wordVec = getWordVector(word)
                countOfWords = countOfWords + 1
                totalDocVec += wordVec
            except:
                continue
    totalDocVec /= countOfWords
    return totalDocVec


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
                plotValue = [wordVec[0],wordVec[1]]
                totalDocVec.append(plotValue)
                correspondingWord.append(word)
            except:
                continue
    return (totalDocVec,correspondingWord)

