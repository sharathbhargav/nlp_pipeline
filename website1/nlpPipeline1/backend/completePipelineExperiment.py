from nlpPipeline1.backend import individualModules as im
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from nlpPipeline1.backend.plotdata import PlottingData
import os
from django.conf import settings



#pathToData="datasets/custom2/"
#pathToPickles = "datasets/custom2/"






def run(fpath, pdir):
    print("Beginning Clustering")
    pathToData = fpath
    pathToPickles = os.path.join(settings.BASE_DIR, pdir)

    custom2Names=open(os.path.join(pathToPickles, 'plotNamesOfDocs'), 'rb')
    fileNames=pickle.load(custom2Names)
    custom2Pickle=open(os.path.join(pathToPickles, 'plotValuesOfDocs'), 'rb')
    total1=pickle.load(custom2Pickle)


    normalized=normalize(total1)


    (clusterCount,clf)=im.customKMeansComplete(normalized)

    print("Clustering done with",clusterCount)
    #labels=clf.labels_
    labels=clf.getLabels(normalized)

    centroids=[]
    centroidsCustom=clf.centroids
    for each in range(len(centroidsCustom)):
        centroid=centroidsCustom[each]
        centroids.append(list(centroid))
    #print(centroids)
    centroids=np.asarray(centroids)
    #print(centroids)
    fileNameDictionary=im.getDocClustersNames(clusterCount,labels,fileNames)
    print("File dict generated")
    for key, val in fileNameDictionary.items():
        fileNameDictionary[key] = [os.path.join(fpath, file) for file in fileNameDictionary[key]]
    file_names = [os.path.join(fpath, file) for file in fileNames]
    pd = PlottingData()
    pd.set_filenames(file_names)
    pd.set_points(normalized)
    pd.set_colors(labels)
    pd.set_clusters(clusterCount, fileNameDictionary, centroids)


    ents = im.getNamedEntties(pathToData,fileNameDictionary,10)
    pd.set_named_entities(ents)


    pd.prepare_to_plot()
    print("Done clustering")

    #im.plotClusters(normalized,fileNames,labels,centroids,True)