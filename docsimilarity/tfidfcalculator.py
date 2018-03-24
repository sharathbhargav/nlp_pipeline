from math import log
from docprocessing.processing import fusion


# Calculates the Term Frequency of the given word in the document
def _tf(word, blob):
    count = 0
    for w in blob.split(' '):
        if w == word:
            count += 1
    return count / len(set(blob))


# Calculates the Inverse Document Frequency of the word in the set of documents
def _idf(word, blob):
    occurences = 0
    for doc in blob:
        if word in doc:
            occurences += 1
    return log(len(blob) / occurences)


# Takes a list of cleaned up documents.
# Calculates tf-idf weight for each word.
# Returns a list of dictionaries, each dictionary corresponding to the input documents.
# The dictionary keys are the words and the value is the tf-idf weight.
def tfidf(blob):
    textblob = fusion(blob)
    tfidfresult = []
    for doc in textblob:
        tfidfdict = dict()
        for word in doc.split(' '):
            tfidfdict[word] = _tf(word, doc) * _idf(word, textblob)
        tfidfresult.append(tfidfdict)
    return tfidfresult