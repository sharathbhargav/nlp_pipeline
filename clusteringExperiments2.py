"""
Goal:
using sklearn implement end to end pipeline
input:Path to set of documents or handles of bunch of documents
output: clusters of documents

process: obtain document vectors -> use PCA to reduce dimensionality -> normalize the vectors using z normalization -> pass the vectors to kmeans,kmeans++, spherical kmeans for clusters 2-20 ->
    obtain silhouette avg values and get best cluster number -> generate clusters with any of the above methods
"""

from gensim.models import Word2Vec
import individualModules as im
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import pickle
from sklearn.decomposition import PCA
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from operator import itemgetter
from algorithms import kMeans as kmeans
from random import randint
from enum import Enum
from sklearn.cluster import Birch

"""
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)

style.use('ggplot')

total1=[]

fileNames=[]


# get document vactors with PCA done and dimension of 2
custom2Pickle=open("datasets/custom2/plotValuesofDocs","rb")
total1=pickle.load(custom2Pickle)
custom2Names=open("datasets/custom2/plotNamesofDocs","rb")
fileNames=pickle.load(custom2Names)

"""
"""

pathSuffix=["b","e","p","s","t"]
for each in pathSuffix:
    pickleFile1=open("datasets/bbc/plotValue"+each,"rb")
    plotValue=pickle.load(pickleFile1)
    pickleFile2=open("datasets/bbc/plotName"+each,"rb")
    plotName=pickle.load(pickleFile2)
    for value in plotValue:
        total1.append(value)
    for name in plotName:
        fileNames.append(name)



#print(normalized)
"""
"""
normalized=normalize(total1)
colors = 100 * ["r", "g", "b", "c", "k"]
"""



#skLearnKMeansComplete()

#customKMeansComplete()