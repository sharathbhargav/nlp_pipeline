from gensim.models import Word2Vec
import individualModules as im
from algorithms import kMeans as kmeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import pickle
from sklearn.decomposition import PCA
import os


#trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
#newsModel= Word2Vec.load("models/newsGroupModel")
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)

style.use('ggplot')

fileHandles1=[]
path1 = "datasets/custom/source1"
fileNmaes=[]
fileCount=0
for fname in os.listdir(path1):
    file = open(os.path.join(path1, fname))
    fileHandles1.append(file)
    fileNmaes.append(fname+str(fileCount))
    fileCount+=1

plotData = im.getPlotValuesOfDocuments(fileHandles1)

total1=np.array(plotData)
#print(total1)
i=0
for k in plotData:
    xy=(k[0],k[1])
    plt.annotate(fileNmaes[i],xy)
    i+=1
#print("Total1",plotData)

colors = 100 * ["r", "g", "b", "c", "k", "l", "p"]
(classifications, centroids) = kmeans.execute_kmeans(total1, k=4, showPlot=False, plotRef=plt)

print(centroids)
count = 0
for centroid in centroids:

    plt.scatter(centroids[centroid][0], centroids[centroid][1], marker="o", color=colors[count], s=100,
                 linewidths=5)
    count = count + 1

for classification in classifications:
    color = colors[classification]
    if len(classifications[classification]) > 0:
        for featureSet in classifications[classification]:
            plt.scatter(featureSet[0], featureSet[1], marker="x", color=color, s=100, linewidths=5)

plt.show()