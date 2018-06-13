from nlpPipeline1.backend.processing import cleanup, wordify, baggify
from numpy import zeros
from .util import gf
from math import log
from sklearn.utils.extmath import randomized_svd
from numpy.matlib import transpose
from numpy import matmul
from .similarity import euclidean_similarity
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir, isabs, abspath
import csv
import os
from django.conf import settings


class TDMatrix:
    def __init__(self, ldocs=None, directory=None, load_from = None, store_to=None):
        self.ldocs = ldocs
        self.n = len(ldocs)
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
        self._generate()

    def _generate(self):
        # print('TDM to CSV')
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
        if len(self.ldocs) <= 300:
            self.dimension = len(self.ldocs)
        else:
            self.dimension = 300
            # print('SVD to CSV')
        self.u, self.sigma, self.vt = randomized_svd(self.tdmatrix, n_components=self.dimension)

    def _get_doc_column(self, index):
        col_matrix = []
        for row in self.tdmatrix:
            col_matrix.append(row[index])
        return col_matrix

    def get_doc_vector(self, doc_index=None, doc_name=None, n_dim=None):
        if n_dim is None:
            n_dim = self.dimension
        if doc_name is None:
            td_column = self._get_doc_column(doc_index)
        elif doc_index is None:
            td_column = self._get_doc_column(self.ldocs.index(doc_name))
        sigma_inverse = [1 / s for s in self.sigma]
        ut = transpose(self.u)
        ut_d = matmul(ut, td_column)
        doc_vector = [sigma_inverse[i] * ut_d[i] for i in range(n_dim)]
        return doc_vector

'''
d = "/home/ullas/PycharmProjects/nlp_pipeline/datasets/bbc/"

import time

st_time = time.time()
lsa = TDMatrix(directory=d, load_from='bbc')
en_time = time.time()
t = en_time - st_time
t /= 60
print("Time taken for model construction = " + str(t) + " minutes")

#st_time = time.time()
#lsa.plot()
#en_time = time.time()
#t = en_time - st_time
#print("Time taken for plotting = " + str(t) + " seconds")
'''


