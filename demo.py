import pickle
import individualModules as im
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA


#trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
newsModel= Word2Vec.load("models/newsGroupModel")
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)

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
    print("Without stop words",getDocumentSimilarityWithoutStopWords(documentHandle1,documentHandle2))
    documentHandle1.seek(0)
    documentHandle2.seek(0)
    ignored=open('documents/ignoredWords','r')
    print("\n","percent ignored",im.getIgnoreWordsPercentage())
    print("With stop words",getDocumentSimilarityWithStopWords(documentHandle1,documentHandle2))
    print("\n","Percent ignored",im.getIgnoreWordsPercentage())
    documentHandle1.seek(0)
    documentHandle2.seek(0)



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


def plotTwoDocs(documentHandle1,documentHandle2):
    (wordVecList1,wordList1)=im.plotDocumentWords(documentHandle1,False)
    (wordVecList2,wordList2)=im.plotDocumentWords(documentHandle2,False)
    plotData1=compressWordVecToPlot(wordVecList1)
    plotData2 = compressWordVecToPlot(wordVecList2)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for k in plotData1:
        x1.append(k[0])
        y1.append(k[1])
    for k in plotData2:
        x2.append(k[0])
        y2.append(k[1])
    plt.scatter(x1,y1,s=10,c="r")
    for i in range(len(wordList1)):
        xy = (x1[i], y1[i])
        plt.annotate(wordList1[i], xy)
    plt.scatter(x2,y2,s=10,c="b")
    for i in range(len(wordList2)):
        xy = (x2[i], y2[i])
        plt.annotate(wordList2[i], xy)
    plt.show()

docTest1 = open("datasets/custom/source1/hyderabadibiryani.txt","r")
docTest2 = open("datasets/custom/source2/hyderabadibiryani.txt","r")
docTest3 = open("datasets/custom/source2/iPhoneX.txt","r")
testDocument = open("documents/testDocument")
printSimilarity(document1,document3)
commonWords=im.getCommonWordsBetweenDocs(document1,document2)


print(commonWords)
print(len(commonWords))
docTest1.seek(0)
docTest2.seek(0)
docTest3.seek(0)
testDocument.seek(0)
#plotDocument(docTest1)
#plotTwoDocs(docTest1,docTest2)
