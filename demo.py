import pickle
import individualModules as im
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA


trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)

document1 = open("documents/news1Hindu","r")
document2 = open("documents/news1NDTV", "r")
document3 = open("documents/news2NDTV","r")
document1.seek(0)
document2.seek(0)
document3.seek(0)

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


def printSimilarity(documentHandle1,documentHandle2):
    print("Without stop words",getDocumentSimilarityWithoutStopWords(document2,document3))
    documentHandle1.seek(0)
    documentHandle2.seek(0)
    print("With stop words",getDocumentSimilarityWithStopWords(document2,document3))
    documentHandle1.seek(0)
    documentHandle2.seek(0)



def compressWordVecToPlot(wordVecList):
    numArray = np.asarray(wordVecList,dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(numArray)
    return pcaOut


def plotDocument(documentHandle,StopWordsRequired=False):
    (wordVecList,wordList) = im.plotDocumentWords(document1,True)
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

plotDocument(document2)
im.getCommonWordsBetweenDocs(document2,document3)