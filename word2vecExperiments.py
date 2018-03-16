import pickle
import individualModules as im
from gensim.models.keyedvectors import KeyedVectors

"""
temp = open("googleTrainedData", "rb")
trainingModelGoogle = pickle.load(temp)
im.setModel(trainingModelGoogle)
"""
trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)

document1 = open("datasets/bbc/politics/001.txt","r")
document2 = open("datasets/bbc/politics/002.txt", "r")
document3 = open("datasets/bbc/entertainment/035.txt","r")

def getDocumentSimilarityWithStopWords(d1,d2):
    docVector1 = im.getDocVector(d1,True)
    docVector2 = im.getDocVector(d2,True)
    docSimilarity = im.getDocSimilarity(docVector1, docVector2)
    return docSimilarity


def getDocumentSimilarityWithoutStopWords(d1,d2):
    docVector1 = im.getDocVector(d1,False)
    docVector2 = im.getDocVector(d2,False)
    docSimilarity = im.getDocSimilarity(docVector1,docVector2)
    return docSimilarity


print("Without stop words",getDocumentSimilarityWithoutStopWords(document1,document3))
document1.seek(0)
document2.seek(0)
document3.seek(0)
print("With stop words",getDocumentSimilarityWithStopWords(document1,document3))