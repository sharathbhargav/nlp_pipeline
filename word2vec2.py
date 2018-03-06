import pickle
import numpy as np
import math
from gensim.models import Word2Vec
import re
import individualModules as im

import nltk
from gensim.models.keyedvectors import KeyedVectors

temp = open("googleTrainedData", "rb")
trainingModelGoogle = pickle.load(temp)

news1File=open("news1Hindu","r")
news2File=open("news1NDTV","r")

im.setModel(trainingModelGoogle)


sentences1 = im.splitCorpusIntoSentances(news1File)
sentences2 = im.splitCorpusIntoSentances(news2File)

docWords1 = []
docWords2 = []

for s1 in sentences1:
    words = im.tokanizeAndRemoveStopWordsSingleSentance(s1)
    docWords1.append(words)

for s2 in sentences2:
    words = im.tokanizeAndRemoveStopWordsSingleSentance(s2)
    docWords2.append(words)

docVector1 = im.getDocVector(docWords1)
docVector2 = im.getDocVector(docWords2)

docSimilarity = im.getDocSimilarity(docVector1,docVector2)
print(docSimilarity)

print("with stop words")
for s1 in sentences1:
    words = im.tokanizeSingleSentance(s1)
    docWords1.append(words)

for s2 in sentences2:
    words = im.tokanizeSingleSentance(s2)
    docWords2.append(words)

docVectorWith1 = im.getDocVector(docWords1)
docVectorWith2 = im.getDocVector(docWords2)

docSimilarityWith1 = im.getDocSimilarity(docVectorWith1,docVectorWith2)
print(docSimilarityWith1)

#print(im.splitCorpusIntoSentances(news1File))


stop = False
while not stop:
    print("Enter two sentences to compare or zzz to exit")
    sent1 = input("Enter sentence one or zzz to exit")
    if "zzz" in sent1:
        stop = True
        break
    sent2 = input("Enter sentence 2")

    print("Cosine similarity between sent1 and sent2")
    print(im.getSentanceSimilarity(sent1, sent2))
