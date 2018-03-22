from docprocessing.processing import cleanup, wordify, baggify
from numpy import zeros
from docsimilarity.util import gf
from math import log
from scipy.linalg import svd
from numpy.matlib import transpose
from numpy import matmul


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
        self.generate()

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
        self._decompose()

    def _decompose(self):
        self.u, self.sigma, self.vt = svd(self.tdmatrix)

    def _get_doc_column(self, index):
        col_matrix = []
        for row in self.tdmatrix:
            col_matrix.append(row[index])
        return col_matrix

    def get_doc_vector(self, doc_index):
        td_column = self._get_doc_column(doc_index)
        sigma_inverse = 1 / self.sigma[doc_index]
        ut = transpose(self.u)
        doc_vector = sigma_inverse * matmul(ut, td_column)
        return doc_vector
