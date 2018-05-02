import numpy as np
from sklearn.decomposition import PCA
from gensim.models.keyedvectors import KeyedVectors
from docprocessing.processing import cleanup

vectorSize = 300
trainingModelGoogle = KeyedVectors.load_word2vec_format("models/GoogleNews-vectors-negative300.bin",binary=True,limit=10000)
modelUsed = trainingModelGoogle


def getSentancesListFromDoc(documentHandle, stopWordsRequired):
    sentances = splitCorpusIntoSentances(documentHandle)
    docWords = []
    for sent in sentances:
        if stopWordsRequired:
            words = tokanizeSingleSentance(sent)
        else:
            words = tokanizeAndRemoveStopWordsSingleSentance(sent)
        if len(words) > 0:
            docWords.append(words)
    return docWords


def getWordVector(inputWord):
    wordVector1 = np.array(modelUsed[inputWord])  # trainingModelGoogle
    return wordVector1


f = open("/home/ullas/PycharmProjects/nlp_pipeline/datasets/reddit/hot/18-04-25_2", 'r')

def getDocVector(documentHandle, stopWordsRequired=False):
    totalDocVec = np.array([float(0.0) for x in range(vectorSize)])
    countOfWords = 0
    countOfIgnore = 0
    completeList = getSentancesListFromDoc(documentHandle, stopWordsRequired)
    ignoredWords = open('documents/ignoredWords', 'w')
    # print(completeList)
    for sentances in completeList:
        # print(sentances)
        for word in sentances:
            try:
                wordVec = getWordVector(word)
                # print("wordvec>>>>>>>>>>>>>>>",wordVec)
                countOfWords = countOfWords + 1
                # print("count>>>>>>>>>>>>>>>>>>>>>",countOfWords)
                totalDocVec += wordVec
                # print(word, ">>>>>>>>", totalDocVec)
            except:
                countOfIgnore += 1

                continue
    totalDocVec /= countOfWords
    ignoredWords.write("Ignored count=" + str(countOfIgnore) + "\n")
    ignoredWords.write("counted=" + str(countOfWords))
    ignoredWords.close()
    return totalDocVec

getDocVector(f)
f.close()


# Takes document handles and returns plotvalues
def getPlotValuesOfDocuments(documentHandles):
    vectors = []
    for handle in documentHandles:
        try:
            vec = getDocVector(handle)
            if (len(vec) > 0):
                vectors.append(vec)
        except:
            print(handle, " failed to read")

    docArray = np.asarray(vectors, dtype=np.float32)
    pca = PCA(n_components=2)
    pcaOut = pca.fit_transform(docArray)
    return pcaOut
