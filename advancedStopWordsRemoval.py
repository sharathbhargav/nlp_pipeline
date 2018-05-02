import individualModules as im
import os
import glob
import pickle
from gensim.models.keyedvectors import KeyedVectors

trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=100000)
im.setModel(trainingModelGoogle)



allExtraWords=[]
def getCommonWords():
    temp=open("datasets/bbc/commonWords.txt","r")
    wordList=im.getSentancesListFromDoc(temp,False)
    print(wordList)
    temp.close()
    temp=open("datasets/bbc/commonWordsPickle","wb")
    pickle.dump(wordList[0],temp)




def openListOfFiles(path):
    fileHandles = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        singleHandle = open(filename,"r")

        fileHandles.append(singleHandle)
    return fileHandles

def closeFileHandles(handleList):
    for i in handleList:
        i.close()

def plotReadyFunc(path):
    handles=openListOfFiles(path)
    getCommonWords(handles)
    #print(values)
    closeFileHandles(handles)





def checkForCommonWords():
    filehandles1= openListOfFiles("datasets/bbc/p")
    filehandles2 = openListOfFiles("datasets/bbc/t")
    totalList=[]
    for i in range(19):
        count=0
        sentances=im.splitCorpusIntoSentances(filehandles1[i])
        #print("Sentances$$$$$$$$$$$$")
        #print(sentances)
        #print("$$$$$$$$$")
        tempCleanedList=[]
        #print("cleaned")
        for sentance in sentances:
            cleaned = im.tokanizeAndRemoveStopWordsSingleSentance(sentance)
            for clean in cleaned:
                tempCleanedList.append(clean)
                count=count+1
        print(i,">>>>>>",count)
        totalList.append(set(tempCleanedList))

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    for i in range(19):
        count=0
        sentances=im.splitCorpusIntoSentances(filehandles2[i])
        #print("Sentances$$$$$$$$$$$$")
        #print(sentances)
        #print("$$$$$$$$$")
        tempCleanedList=[]
        #print("cleaned")
        for sentance in sentances:
            cleaned = im.tokanizeAndRemoveStopWordsSingleSentance(sentance)
            for clean in cleaned:
                tempCleanedList.append(clean)
                count=count+1
        print(i,">>>>>>",count)
        totalList.append(set(tempCleanedList))



    sum=0
    for i in range(19):
        for j in range(19):
            commonWords = totalList[i].intersection(totalList[19+j])
            sum=sum+len(commonWords)
            if i != j:
                print(i," and  ",(20+j),"==",len(commonWords))

    print("average=",sum/400)

#checkForCommonWords()

path1="datasets/bbc/politics"
path2= "datasets/bbc/tech"
path3 = "datasets/bbc/sport"
path4 = "datasets/bbc/business"
path5 = "datasets/bbc/entertainment"



temp=open("datasets/bbc/commonWordsPickle","rb")
allExtraWords=pickle.load(temp)
#print(allExtraWords)

def writeNewFiles(path1,path2):

    readhandlers = list()
    for file in os.listdir(path1):
        filename = path1 + '/' + file
        readhandlers.append(open(filename, 'r'))

    writehandles = list()
    for file in os.listdir(path2):
        filename = path2 + '/' + file
        writehandles.append(open(filename, 'w+'))

    for i in range(len(readhandlers)):
        handle=readhandlers[i]
        wordList=im.getSentancesListFromDoc(handle,False)
        #print(wordList)

        cleanedDoc=[]
        for sentances in wordList:
            for word in sentances:
                if word not in allExtraWords:
                    cleanedDoc.append(word)

        writeHandle1=writehandles[i]

        writeHandle1.write(" ".join(cleanedDoc))

    closeFileHandles(readhandlers)
    closeFileHandles(writehandles)

#writeNewFiles(path2,"datasets/bbc/t")
#writeNewFiles(path3,"datasets/bbc/s")
#writeNewFiles(path4,"datasets/bbc/b")
#writeNewFiles(path5,"datasets/bbc/e")