import pickle
import individualModules as im
from gensim.models.keyedvectors import KeyedVectors
"""
temp = open("googleTrainedData", "rb")
trainingModelGoogle = pickle.load(temp)
im.setModel(trainingModelGoogle)
"""
trainingModelGoogle = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)

document1 = open("documents/news1Hindu","r")
document2 = open("documents/news1NDTV", "r")
document3 = open("documents/news2NDTV","r")

def getDocumentSimilarityWithStopWords(d1,d2):
    sentences1 = im.splitCorpusIntoSentances(d1)
    sentences2 = im.splitCorpusIntoSentances(d2)
    docWords1 = []
    docWords2 = []
    for s1 in sentences1:
        words = im.tokanizeSingleSentance(s1)
        docWords1.append(words)

    for s2 in sentences2:
        words = im.tokanizeSingleSentance(s2)
        docWords2.append(words)

    docVector1 = im.getDocVector(docWords1)
    docVector2 = im.getDocVector(docWords2)
    docSimilarity = im.getDocSimilarity(docVector1, docVector2)
    return docSimilarity



