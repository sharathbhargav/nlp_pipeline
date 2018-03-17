from docprocessing.processing import cleanup, wordify, baggify
from numpy import zeros
from docsimilarity.util import gf
from math import log
from scipy.linalg import svd


class TDMatrix:
    def __init__(self, ldocs, directory=""):
        if len(ldocs) > 0:
            # ldocs is a list of documents
            self.ldocs = ldocs
            self.n = len(ldocs)
        else:
            print("Expected more documents")
        if directory != "":
            # directory containing all the files listed in ldocs.
            self.directory = directory
            for i in range(0, len(self.ldocs)):
                self.ldocs[i] = directory + '/' + self.ldocs[i]
        # doc_sentences : List of documents in the form of cleaned sentences
        self.doc_sentences = list()
        for doc in self.ldocs:
            with open(doc, 'r') as f:
                self.doc_sentences.append(cleanup(f))
        # doc_words : List of documents, each represented as as a list of words.
        self.doc_words = wordify(self.doc_sentences)
        # bag_of_words : List of documents, each represented as a bag of unique words.
        self.bag_of_words = baggify(self.doc_words)
        # tdmatrix : Term Document matrix for the set of documents.
        self.tdmatrix = zeros(shape=(len(self.bag_of_words), len(self.ldocs)))

    def generate(self):
        g = dict()
        # Calculating global weights for each word
        for iword, word in enumerate(self.bag_of_words):
            gsigma = 0
            for jdoc, doc in enumerate(self.ldocs):
                if word in self.doc_words[jdoc]:
                    pij = self.doc_words[jdoc].count(word) / gf(word, self.doc_words)
                    gsigma += (pij * log(pij)) / log(self.n)
            g[word] = 1 + gsigma
        # Creating the Term-Document matrix.
        # Local weight is log weight, l = log(1 + tf_ij)
        # Global weight is entropy weight, g_i = 1 + sigma((p_ij * log p_ij) / log n)
        # p_ij = tf_ij / gf_i
        for iword, word in enumerate(self.bag_of_words):
            for jdoc, doc in enumerate(self.ldocs):
                if word in self.doc_words[jdoc]:
                    self.tdmatrix[iword][jdoc] = g[word] * log(self.doc_words[jdoc].count(word) + 1)
        print('A: ' + str(len(self.tdmatrix)) + ' x ' + str(len(self.tdmatrix[0])))

    def decompose(self):
        self.u, self.sigma, self.vt = svd(self.tdmatrix)
        print('U: ' + str(len(self.u)) + ' x ' + str(len(self.u[0])))
        print(self.sigma)
        print('Vt: ' + str(len(self.vt)) + ' x ' + str(len(self.vt[0])))





f1 = "/home/ullas/nltk_trial/corpora/hp/ff_ootp.txt"
f2 = "/home/ullas/nltk_trial/corpora/eragon/eldest.txt"
f3 = "/home/ullas/nltk_trial/corpora/eragon/brisingr.txt"
f4 = "/home/ullas/nltk_trial/corpora/eragon/eragon.txt"

a = TDMatrix([f1, f2, f3, f4])
a.generate()
a.decompose()