import os
from website.doccer.pipeline import individualModules as im
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

path1 = "datasets/custom/source1"
path2 = "datasets/custom/source2"
fileHandles1=[]
fileHandles2=[]



trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
newsModel= Word2Vec.load("models/newsGroupModel")
medicalModel = Word2Vec.load("models/medicalModel")
im.setModel(medicalModel)
for fname in os.listdir(path1):
    file = open(os.path.join(path1, fname))
    fileHandles1.append(file)

for fname in os.listdir(path2):
    file = open(os.path.join(path2, fname))
    fileHandles2.append(file)

i=0
for fname in os.listdir(path1):
    docVec1 = im.getDocVector(fileHandles1[i],False)
    docVec2= im.getDocVector(fileHandles2[i],False)
    docSim = im.getDocSimilarity(docVec1,docVec2)
    print(fname,">>",docSim)
    i+=1
