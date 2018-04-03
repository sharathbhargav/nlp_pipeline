import pickle
import individualModules as im
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
"""
temp = open("googleTrainedData", "rb")
trainingModelGoogle = pickle.load(temp)
im.setModel(trainingModelGoogle)
"""
#trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
trainingModel=Word2Vec.load("models/medicalModel")
im.setModel(trainingModel)

document1 = open("documents/news1Hindu","r")
document2 = open("documents/news1NDTV", "r")
document3 = open("documents/news2NDTV","r")
document4 = open("datasets/bbc/b/001.txt","r")
document5 = open("datasets/bbc/b/002.txt","r")
document1.seek(0)
document2.seek(0)
document3.seek(0)
document4.seek(0)
document5.seek(0)

def getDocumentSimilarityWithStopWords(d1,d2):
    docVector1 = im.getDocVector(d1,False)
    docVector2 = im.getDocVector(d2,False)
    docSimilarity = im.getDocSimilarity(docVector1, docVector2)
    return docSimilarity


def getDocumentSimilarityWithoutStopWords(d1,d2):
    docVector1 = im.getDocVector(d1,False)
    docVector2 = im.getDocVector(d2,False)
    docSimilarity = im.getDocSimilarity(docVector1,docVector2)
    return docSimilarity

#print(im.getDocVector(document1,False))

print("Without stop words",getDocumentSimilarityWithoutStopWords(document2,document3))
document1.seek(0)
document2.seek(0)
document3.seek(0)
document4.seek(0)
document5.seek(0)

print("With stop words",getDocumentSimilarityWithStopWords(document2,document3))
document1.seek(0)
document2.seek(0)
document3.seek(0)
document4.seek(0)
document5.seek(0)
im.getCommonWordsBetweenDocs(document2,document3)