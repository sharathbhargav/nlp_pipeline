import numpy as np
from . import individualModules as im
import matplotlib.pyplot as plot
from matplotlib import style

style.use("ggplot")

testData2 = np.array([
    [0.02005346, 0.04187598],
    [0.0269297, 0.03662949],
    [0.0530976, 0.03613571],
    [0.0225159, 0.04469353],
    [0.02389053, 0.03895409],
    [0.02834501, 0.02974286],
    [0.01130847, 0.04768526],
    [0.00978073, 0.02735387]
])
testData3 = np.array([
    [0.8713264, 1.6529288],
    [-0.13857056, -0.45318842],
    [-1.070514, 0.36548918],
    [-1.0965503, 0.04482369],
    [0.58449924, 1.013452],
    [0.10820448, -0.53374726],
    [1.9872766, -0.87119573],
    [-0.2862797, -0.87349224],
    [-0.9708758, 0.11404149],
    [0.01148286, -0.4591116]
])

colors = 10 * ["r", "g", "b", "c", "k"]


# plot.scatter(testData[:,0],testData[:,1],s=15)
# plot.show()

class K_Means:
    def __init__(self, k=5, tolerance=0.00001, max_iterations=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iterations

    def getDistance(self, vec1, vec2, spherical=False):
        if spherical:
            return 1 - im.getDocSimilarity(vec1, vec2)
        else:
            return im.getDocSimilarity(vec1, vec2)

    def fit(self, data, spherical=False, rotationArray=[]):
        self.centroids = {}
        #print(np.asarray([5.01, 6.01], dtype=float))
        """
        randomData=[]
        for r in range(self.k):
            random=r+1
            random1=[random*0.0256,random*0.0885]
            randomData.append(random1)
        randomArray=np.array(randomData)
        """

        for i in range(self.k):
            self.centroids[i] = rotationArray[i]
        #print(self.centroids)
        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []
            # print("i>>>>>>>>>>>>>>>", i)
            # print("Centroids",self.centroids)
            for featureset in data:
                #print(featureset)
                distances = []
                for centroid in self.centroids:
                    # print(">>>>>>>>>>>individual",centroid,featureset)
                    distance = self.getDistance(featureset, self.centroids[centroid], spherical)
                    distances.append(distance)
                # distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # print(distances)
                classification = distances.index(min(distances))
                #print("classi>>>>>>", classification)
                self.classifications[classification].append(featureset)

            #print("classificatin>>>>>>>>>>>",self.classifications)
            prevCentroids = dict(self.centroids)
            for classification in self.classifications:

                if(len(self.classifications[classification])>1):
                    avg=np.average(self.classifications[classification],axis=0)
                    self.centroids[classification] = avg
                elif(len(self.classifications[classification])==1):
                    self.centroids[classification] = self.classifications[classification]

                # print("Avg>>>>>>>>",self.classifications[classification])
                #self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            #print(">>>>>>>>>>>", self.centroids)
            optimized = True

            for c in self.centroids:
                originalCentroids = prevCentroids[c]
                currentCentroid = self.centroids[c]
                if np.sum((currentCentroid - originalCentroids) / originalCentroids * 100.0) > self.tolerance:
                    optimized = False

            if optimized:
                break

    def getLabels(self,data):
        labels=[]
        #print(self.classifications)
        for entity in range(len(data)):

            for each in range(len(self.classifications)):
                for classification in self.classifications[each]:
                    if all(data[entity] ==classification):
                        labels.append(each)
        return labels




    def predict(self, data):
        distances = [self.getDistance(data, self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification




def execute_kmeans(data, k=3, sphericalDistance=False, showPlot=False, tolerance=0.0001, max_iterations=300,
                   plotRef=plot,rotationArray=[]):

        clf = K_Means(k, tolerance=tolerance, max_iterations=max_iterations)
        # print("predict", clf.predict(data))

        try:

            clf.fit(data, sphericalDistance, rotationArray)

        except:

            raise Exception("Mean exception")

        print(clf.getLabels(data))
        for iteration in range(len(clf.centroids)):
            if abs(np.mean(clf.centroids[iteration]))>0:
                continue
            else:
                raise Exception("Stupid cluster")
        if showPlot:

            count = 0
            for centroid in clf.centroids:
                plotRef.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color=colors[count],
                                s=100,
                                linewidths=5)
                count = count + 1

            for classification in clf.classifications:
                color = colors[classification]
                if len(clf.classifications[classification]) > 0:
                    for featureSet in clf.classifications[classification]:
                        plotRef.scatter(featureSet[0], featureSet[1], marker="x", color=color, s=100, linewidths=5)

            # plotRef.show()
        return (clf.classifications, clf.centroids)




#(c1,c2)=execute_kmeans(testData3,k=3,sphericalDistance=True,rotationArray=[testData3[0],testData3[1],testData3[2]])

# print(c1)
# print(c2)


# print(testData3)