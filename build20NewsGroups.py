import psutil
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import pickle
import datetime
import  logging
import re
from pyspectator import temperature_reader as tempReader
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stops = set(stopwords.words('english'))

punctuations = ['.', ',', '/', '<', '>', '?', ';', '\'', ':', '"', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^',
                '&', '*', '(', ')', '-', '_', '=', '+', '—']

stops.update(punctuations)
parentPath="datasets/20_newsgroups"


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
            try:
                sentList=sent_tokenize(file.read())
                for lines in sentList:
                    words=word_tokenize(lines)
                    tempWords= [word.lower() for word in words if word.lower() not in stops]
                    yield tempWords
            except:
                print("invalid")
"""

newsGroupModel=Word2Vec(size=300)

trainList=[]
for fname in os.listdir(parentPath):
    pathOfEach = os.path.join(parentPath,fname)
    #sentances1= MySentences(pathOfEach)
    sentances2 = MySentences(pathOfEach)
    #vocabList.append(sentances1)
    trainList.append(sentances2)


vocabFile=open("datasets/NewsvocabFile","rb")
totalVocab=pickle.load(vocabFile)


print(datetime.datetime.time(datetime.datetime.now()))
newsGroupModel.build_vocab(totalVocab)
print("Vocab building done with ",newsGroupModel.corpus_count)
print(datetime.datetime.time(datetime.datetime.now()))
count=0
for train in trainList:
    print(count)
    count+=1
    newsGroupModel.train(train,total_examples=newsGroupModel.corpus_count,epochs=newsGroupModel.iter)
    print(psutil.virtual_memory())
    print(psutil.cpu_percent(interval=1,percpu=True))

print(datetime.datetime.time(datetime.datetime.now()))
newsGroupModel.save("models/newsGroupModel")
"""