from gensim.models import Word2Vec
from website.doccer.pipeline import individualModules as im
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle

#trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
#newsModel= Word2Vec.load("models/newsGroupModel")
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)

style.use('ggplot')

fileHandles1=[]
pathSuffix=["Kathua","Sci","Sports","Tech"]
path1 = "datasets/custom2/Kathua"
pathPrefix="datasets/custom2/"
fileNames=[]
fileCount=0
"""
for i in range(4):
    path2=pathPrefix+pathSuffix[i]

    for fname in os.listdir(path2):

        file = open(os.path.join(path2, fname))
        fileHandles1.append(file)
        fileNmaes.append(pathSuffix[i]+str(fileCount))
        fileCount+=1

plotData = im.getPlotValuesOfDocuments(fileHandles1)

total1=np.array(plotData)
"""
custom2Pickle=open("datasets/custom2/plotValuesofDocs","rb")

total1=pickle.load(custom2Pickle)
custom2Names=open("datasets/custom2/plotNamesofDocs","rb")
fileNames=pickle.load(custom2Names)
#print(total1)

#print("Total1",plotData)





def errorCalculation(centroids,classifications):
    squaredSum=0
    for i in range(len(centroids)):
        centroidPoint=centroids[i]
        for each in classifications[i]:

            diff=np.linalg.norm(centroidPoint-each)
            #print(diff)
            squaredSum=squaredSum+(diff*diff)
    return squaredSum
"""

i=0
for k in total1:
    xy=(k[0],k[1])
    #plt.scatter(k[0],k[1],color=colors[labelsKmeans[i]],marker="o",s=25,linewidths=5)
    plt.annotate(fileNames[i],xy)
    i+=1
colors = 100 * ["r", "g", "b", "c", "k", "l", "p"]

totalClusters=3
for iteration in range(1):#len(total1)-totalClusters):
    try:
        rotationArray = []
        rStart = 1

        for r in range(totalClusters):
            rotationArray.append(total1[rStart])
            rStart += 1
        (classifications, centroids) = kmeans.execute_kmeans(total1, k=totalClusters,sphericalDistance=True, showPlot=False, plotRef=plt,rotationArray=rotationArray)

        #print(centroids)
        #print(">>>>>>>>>>>")
        #print(classifications)
        error=errorCalculation(centroids,classifications)
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
        
    except:
        print(iteration)
"""
errorTable1={}
for clusterNumber in range(2,20):

    rotationArray = []
    rStart = 25
    for r in range(clusterNumber):
            rotationArray.append(total1[rStart])
            rStart += 1
    print(clusterNumber)


    try:
        (classifications, centroids) = kMeans.execute_kmeans(total1, k=clusterNumber, showPlot=False, plotRef=plt,
                                                             rotationArray=rotationArray)

        #print(centroids)
        error=errorCalculation(centroids,classifications)
        errorTable1[clusterNumber]=error

    except:
        print("clustering failed at",clusterNumber)
print(errorTable1)
plt.plot(errorTable1.keys(),errorTable1.values())
plt.show()


""""
print(labelsKmeans)
i=0
for k in total1:
    xy=(k[0],k[1])
    plt.scatter(k[0],k[1],color=colors[labelsKmeans[i]],marker="o",s=25,linewidths=5)
    plt.annotate(fileNames[i],xy)
    i+=1
plt.scatter(centroidsKmeans[:,0],centroidsKmeans[:,1],marker='x',s=150,linewidths=5)
plt.show()
"""
