import pickle
import os
import individualModules as im
from gensim.models import Word2Vec
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


#trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
#im.setModel(trainingModelGoogle)

"""
pathSuffix=["b","e","p","s","t"]

pathPrefix="datasets/bbc/"
fileNames=[]
fileCount=0

for i in range(len(pathSuffix)):
    path2=pathPrefix+pathSuffix[i]
    fileHandles1 = []
    fileNames = []
    for fname in os.listdir(path2):
        file = open(os.path.join(path2, fname))
        fileHandles1.append(file)
        fileNames.append(pathSuffix[i]+str(fileCount))
        fileCount+=1
    plotData = im.getPlotValuesOfDocuments(fileHandles1)
    total=np.array(plotData)
    pickleFile=open("datasets/bbc/plotValue"+pathSuffix[i],"wb")
    pickleFile2=open("datasets/bbc/plotName"+pathSuffix[i],"wb")
    pickle.dump(total,pickleFile)
    pickle.dump(fileNames,pickleFile2)
"""

fileHandles1=[]
pathSuffix=["Kathua","Sci","Sports","Tech"]

pathPrefix="datasets/custom2/"
fileNames=[]
fileCount=0

for i in range(4):
    path2=pathPrefix+pathSuffix[i]

    for fname in os.listdir(path2):

        file = open(os.path.join(path2, fname))
        fileHandles1.append(file)
        fileNames.append(pathSuffix[i]+"/"+fname)
        fileCount+=1

plotData = im.getPlotValuesOfDocuments(fileHandles1)

total1=np.array(plotData)

custom2Pickle=open("datasets/custom2/plotValuesOfDocs","wb")

pickle.dump(total1,custom2Pickle)
custom2Names=open("datasets/custom2/plotNamesOfDocs","wb")
pickle.dump(fileNames,custom2Names)

"""
fileNames=[]
fileCount=0
path2="/media/sharathbhragav/New Volume/redditPosts/hot"
fileHandles1=[]

for fname in os.listdir(path2):
    file = open(os.path.join(path2, fname))
    fileHandles1.append(file)
    fileNames.append(fname)
    fileCount+=1

plotData = im.getPlotValuesOfDocuments(fileHandles1)

total1=np.array(plotData)

custom2Pickle=open("/media/sharathbhragav/New Volume/redditPosts/pickles/plotValuesOfDocs","wb")

pickle.dump(total1,custom2Pickle)
custom2Names=open("/media/sharathbhragav/New Volume/redditPosts/pickles/plotNamesOfDocs","wb")
pickle.dump(fileNames,custom2Names)

"""