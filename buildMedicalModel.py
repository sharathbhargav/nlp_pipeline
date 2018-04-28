from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import pickle
import datetime
import psutil
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stops = set(stopwords.words('english'))

punctuations = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—']

stops.update(punctuations)

file=open('datasets/bbc/b/130.txt')
print(file.read())
file.close()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        """
        for fname in os.listdir(self.dirname):
            file = open(fname)
            print(file)
            line=file.readline()
            file.close()
            return line
        """
        for fname in os.listdir(self.dirname):
            file=open(os.path.join(self.dirname,fname))
            for lines in sent_tokenize(file.read()):
                words=word_tokenize(lines)
                tempWords= [word.lower() for word in words if word.lower() not in stops]
                yield tempWords

#taining medical model
"""
medicalModel=Word2Vec(size=300)
vocabList=[]
trainList=[]
for i in range(1,10):

    #sentances1 = MySentences('datasets/medical/C0'+str(i))
    #vocabList.append(sentances1)
    sentances2 = MySentences('datasets/medical/C0' + str(i))
    #medicalModel.train(sentances2,total_examples=medicalModel.corpus_count,epochs=medicalModel.iter)
    trainList.append(sentances2)
for i in range(10,24):
    #sentances1 = MySentences('datasets/medical/C' + str(i))
    #vocabList.append(sentances1)
    sentances2 = MySentences('datasets/medical/C' + str(i))
    trainList.append(sentances2)



vocabFile=open("datasets/medical/vocabFile","rb")


for vocabs in vocabList:
    for vocab in vocabs:

        totalVocab.append(vocab)
    print(psutil.virtual_memory())
    print(psutil.cpu_percent(interval=1, percpu=True))

pickle.dump(totalVocab,vocabFile)
print(len(totalVocab))
count=0

totalVocab=pickle.load(vocabFile)
medicalModel.build_vocab(totalVocab)
print("build vocab done")
count=0
for train in trainList:
    print(count)
    count+=1
    medicalModel.train(train,total_examples=medicalModel.corpus_count,epochs=medicalModel.iter)
    print(psutil.virtual_memory())
    print(psutil.cpu_percent(interval=1, percpu=True))


medicalModel.save("models/medicalModel")

medicalModel=Word2Vec.load("models/medicalModel1")
print(medicalModel['death'])
print(len(medicalModel.wv.vocab))
"""
