from math import log, pow, sqrt
from docprocessing.processing import fusion, cleanup, wordify
from docprocessing.util import get_files


class TFIDFVectorizer:
    # files: List of file names with absolute path
    # bag_of_words: List of list of words
    # tfidfweights: List of dictionaries with tfidf weights
    def __init__(self, dirname):
        self.files = get_files(dirname)
        handlers = [open(file, 'r') for file in self.files]
        self.cleaned_files = [cleanup(f) for f in handlers]
        self.bag_of_words = wordify(self.cleaned_files)
        self.tfidfweights = self.tfidf()

    # Calculates the Term Frequency of the given word in the document
    def _tf(self, word, blob):
        count = 0
        for w in blob.split(' '):
            if w == word:
                count += 1
        return count / len(set(blob))

    # Calculates the Inverse Document Frequency of the word in the set of documents
    def _idf(self, word, blob):
        occurences = 0
        for doc in blob:
            if word in doc:
                occurences += 1
        return log(len(blob) / occurences)

    # Takes a list of cleaned up documents.
    # Calculates tf-idf weight for each word.
    # Returns a list of dictionaries, each dictionary corresponding to the input documents.
    # The dictionary keys are the words and the value is the tf-idf weight.
    def tfidf(self):
        textblob = fusion(self.cleaned_files)
        tfidfresult = []
        for doc in textblob:
            tfidfdict = dict()
            for word in doc.split(' '):
                tfidfdict[word] = self._tf(word, doc) * self._idf(word, textblob)
            tfidfresult.append(tfidfdict)
        return tfidfresult

    # Calculates the Euclidean distance between the two files in the given indices
    def distance(self, fi1, fi2):
        if fi1 > len(self.files):
            print("Error. Index 1 out of range")
            return None
        elif fi2 > len(self.files):
            print("Error. Index 2 out of range")
            return None
        print("File 1 : " + self.files[fi1])
        print("File 2 : " + self.files[fi2])
        bag_1 = self.bag_of_words[fi1]
        bag_2 = self.bag_of_words[fi2]
        term_set = set(bag_1).intersection(set(bag_2))
        vector_1 = [self.tfidfweights[fi1][word] for word in term_set]
        vector_2 = [self.tfidfweights[fi2][word] for word in term_set]
        _sq_sum = 0
        for i in range(len(vector_1)):
            _sq_sum += pow((vector_2[i] - vector_1[i]), 2)
        e_dist = sqrt(_sq_sum)
        return e_dist