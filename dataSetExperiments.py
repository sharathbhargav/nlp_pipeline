import os
import glob
import individualModules as im
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import pickle

from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors

style.use('ggplot')

trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)

def getDocumentsValues(listOfHandles):
    vectors=[]
    for handle in listOfHandles:
        vec=im.getDocVector(handle)
        vectors.append(vec)

    values=[]
    for vec in vectors:
        sum=0
        values.append([vec[0],vec[1]])
    return values

def openListOfFiles(path):
    fileHandles = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        singleHandle = open(filename,"r")
        fileHandles.append(singleHandle)
    return fileHandles

def closeFileHandles(handleList):
    for i in handleList:
        i.close()

def plotReadyFunc(path):
    handles=openListOfFiles(path)
    values= getDocumentsValues(handles)
    closeFileHandles(handles)
    return values

"""
DocumentWordVectorValues=im.plotDocumentWords(fileHandles[2])
print(DocumentWordVectorValues)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(DocumentWordVectorValues,"bo")
plt.title("Word Vectors")
plt.show()
fileHandles[2].seek(0)
"""



#reading files and storing first two values of each document
"""
path1 = "datasets/bbc/politics"
path2 = "datasets/bbc/business"
path3 = "datasets/bbc/tech"
path4 = "datasets/bbc/entertainment"
path5 = "datasets/bbc/sport"

plot1=np.array(plotReadyFunc(path1))
plot2= np.array(plotReadyFunc(path2))
plot3=np.array(plotReadyFunc(path3))
plot4 = np.array(plotReadyFunc(path4))
"""

valueDumpFile1 = open("datasets/bbc/politics.pickle","rb")
plot1=pickle.load(valueDumpFile1)

valueDumpFile1 = open("datasets/bbc/business.pickle","rb")
plot2=pickle.load(valueDumpFile1)

valueDumpFile1 = open("datasets/bbc/tech.pickle","rb")
plot3=pickle.load(valueDumpFile1)

valueDumpFile1 = open("datasets/bbc/entertainment.pickle","rb")
plot4=pickle.load(valueDumpFile1)

print(plot1)
fig = plt.figure()
ax = plt.subplot(111)
"""
clf= KMeans(n_clusters=4)
clf.fit(plot1)

clusters = clf.cluster_centers_
labels = clf.labels_
plt.scatter(clusters[:,0],clusters[:,1],marker='X',size=5)

"""
ax.scatter(plot1[:,0],plot1[:,1],s=5,linewidths=5)
#line1=ax.plot(plot1[:,0],plot1[:,1],"bo",label="Politics")
#line2=ax.plot(plot2[:,0],plot2[:,1],"ro",label='Business')
#line3=ax.plot(plot3[:,0],plot3[:,1],"go",label="Tech")
#line4=plt.plot(plot4[:,0],plot4[:,1],"yo",label="Entertainment")
ax.legend()
fig.add_subplot(ax)
#fig.add_subplot(aq)
plt.show()

