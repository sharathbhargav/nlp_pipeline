from docprocessing.processing import cleanup, wordify, baggify
from numpy import zeros
from docsimilarity.util import gf
from math import log
from sklearn.utils.extmath import randomized_svd
from numpy.matlib import transpose
from numpy import matmul
from docsimilarity.similarity import euclidean_similarity
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir, isabs, abspath
from os import PathLike
import csv


class TDMatrix:
    def __init__(self, ldocs=None, directory=None, load_from = None, store_to=None):
        self.load_from = load_from
        self.store_to = store_to
        if directory is None and ldocs is not None:
            print('1')
            self.ldocs = ldocs
            self.n = len(self.ldocs)
            d = ldocs[0].split('/')
            dd = '/'
            for dname in d[:-1]:
                dd = dd + dname + '/'
            self.directories = [dd]
        elif directory is not None and ldocs is not None:
            print('2')
            self.ldocs = [directory + '/' + doc for doc in ldocs]
            self.directories = [directory]
            self.n = len(self.ldocs)
        elif directory is not None and ldocs is None:
            print('3')
            self.directories = list()
            self.ldocs = list()
            files = listdir(directory)
            for file in files:
                file = directory + file + '/'
                if isdir(file):
                    self.directories.append(file)
                    for filename in listdir(file):
                        filename = file + filename
                        self.ldocs.append(filename)
            self.n = len(self.ldocs)

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
        if self.load_from is None and self.store_to is not None:
            print('TDM to CSV')
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
            print("TDM Done")
            csv_filename = "../models/" + self.store_to
            csv_file = open(csv_filename + '_tdmatrix.csv', 'w+')
            writer = csv.writer(csv_file, delimiter=',')
            for row in self.tdmatrix:
                writer.writerow(row)
            csv_file.close()
        elif self.load_from is not None and self.store_to is None:
            print('TDM from CSV')
            csv_filename = "../models/" + self.load_from
            csv_file = open(csv_filename + '_tdmatrix.csv', 'r')
            reader = csv.reader(csv_file)
            for irow, row in enumerate(reader):
                for icol, val in enumerate(row):
                    self.tdmatrix[irow][icol] = val
            csv_file.close()
        self._decompose()

    def _decompose(self):
        if len(self.ldocs) <= 300:
            self.dimension = len(self.ldocs)
        else:
            self.dimension = 300
        if self.load_from is None and self.store_to is not None:
            print('SVD to CSV')
            self.u, self.sigma, self.vt = randomized_svd(self.tdmatrix, n_components = self.dimension)
            csv_filename = "../models/" + self.store_to
            csv_u = open(csv_filename + '_u.csv', 'w+')
            csv_sigma = open(csv_filename + '_sigma.csv', 'w+')
            csv_vt = open(csv_filename + '_vt.csv', 'w+')
            writer_u = csv.writer(csv_u, delimiter=',')
            writer_sigma = csv.writer(csv_sigma, delimiter=',')
            writer_vt = csv.writer(csv_vt, delimiter=',')

            for row in self.u:
                writer_u.writerow(row)
            writer_sigma.writerow(self.sigma)
            for row in self.vt:
                writer_vt.writerow(row)
            csv_u.close()
            csv_sigma.close()
            csv_vt.close()
        elif self.load_from is not None and self.store_to is None:
            print('SVD from CSV')
            csv_filename = "../models/" + self.load_from
            csv_u = open(csv_filename + '_u.csv', 'r')
            csv_sigma = open(csv_filename + '_sigma.csv', 'r')
            csv_vt = open(csv_filename + '_vt.csv', 'r')
            reader_u = csv.reader(csv_u)
            reader_sigma = csv.reader(csv_sigma)
            reader_vt = csv.reader(csv_vt)
            self.u = list()
            for irow, row in enumerate(reader_u):
                self.u.append(list())
                for icol, val in enumerate(row):
                    self.u[irow].append(val)
            self.sigma = list()

            for val in reader_sigma:
                self.sigma.append(val)

            self.vt = list()
            for irow, row in enumerate(reader_vt):
                self.vt.append(list())
                for icol, val in enumerate(row):
                    self.vt[irow].append(val)
            csv_u.close()
            csv_sigma.close()
            csv_vt.close()
        print("SVD Done")
        print("X : " + str(len(self.tdmatrix)) + " x " + str(len(self.tdmatrix[0])))
        print("U : " + str(len(self.u)) + " x " + str(len(self.u[0])))
        print("Sigma : " + str(len(self.sigma)))
        print("Vt : " + str(len(self.vt)) + " x " + str(len(self.vt[0])))

    def _get_doc_column(self, index):
        col_matrix = []
        for row in self.tdmatrix:
            col_matrix.append(row[index])
        return col_matrix

    def get_doc_vector(self, doc_index, n_dim=None):
        if n_dim is None:
            n_dim = self.dimension
        td_column = self._get_doc_column(doc_index)
        if n_dim is None:
            sigma_inverse = [1 / s for s in self.sigma]
            ut = transpose(self.u)
        else:
            sigma_inverse = [1 / s for s in self.plot_sigma]
            ut = transpose(self.plot_u)
        ut_d = matmul(ut, td_column)
        doc_vector = [sigma_inverse[i] * ut_d[i] for i in range(n_dim)]
        return doc_vector

    def plot(self):
        colors = ['b', 'g', 'r', 'c', 'm']
        self.plot_axes = 2
        if self.load_from is None and self.store_to is not None:
            print('Plot to CSV')
            self.plot_u, self.plot_sigma, self.plot_vt = randomized_svd(self.tdmatrix, n_components=self.plot_axes)
            csv_filename = "../models/" + self.store_to
            csv_u = open(csv_filename + '_plot_u.csv', 'w+')
            csv_sigma = open(csv_filename + '_plot_sigma.csv', 'w+')
            csv_vt = open(csv_filename + '_plot_vt.csv', 'w+')
            writer_u = csv.writer(csv_u, delimiter=',')
            writer_sigma = csv.writer(csv_sigma, delimiter=',')
            writer_vt = csv.writer(csv_vt, delimiter=',')

            for row in self.plot_u:
                writer_u.writerow(row)
            writer_sigma.writerow(self.plot_sigma)
            for row in self.plot_vt:
                writer_vt.writerow(row)
            csv_u.close()
            csv_sigma.close()
            csv_vt.close()
        elif self.load_from is not None and self.store_to is None:
            print('Plot from CSV')
            csv_filename = "../models/" + self.load_from
            csv_u = open(csv_filename + '_plot_u.csv', 'r')
            csv_sigma = open(csv_filename + '_plot_sigma.csv', 'r')
            csv_vt = open(csv_filename + '_plot_vt.csv', 'r')
            reader_u = csv.reader(csv_u)
            reader_sigma = csv.reader(csv_sigma)
            reader_vt = csv.reader(csv_vt)
            self.plot_u = list()
            for irow, row in enumerate(reader_u):
                self.plot_u.append(list())
                for icol, val in enumerate(row):
                    self.plot_u[irow].append(val)

            self.plot_sigma = list()
            for val in reader_sigma:
                self.plot_sigma.append(val)

            self.plot_vt = list()
            for irow, row in enumerate(reader_vt):
                self.plot_vt.append(list())
                for icol, val in enumerate(row):
                    self.plot_vt[irow].append(val)
            csv_u.close()
            csv_sigma.close()
            csv_vt.close()

        print("Plot SVD Done")
        print("U : " + str(len(self.plot_u)) + " x " + str(len(self.plot_u[0])))
        print("Sigma : " + str(len(self.plot_sigma)))
        print("Vt : " + str(len(self.plot_vt)) + " x " + str(len(self.plot_vt[0])))
        vectors = list()
        for idir, dirname in enumerate(self.directories):
            vectors.append(list())
            for file in listdir(dirname):
                index = self.ldocs.index(dirname + file)
                vectors[idir].append(self.get_doc_vector(index, self.plot_axes))
        X = list()
        Y = list()
        for i in range(len(vectors)):
            X.append(list())
            Y.append(list())
            for vec in vectors[i]:
                X[i].append(vec[0])
                Y[i].append(vec[1])

        for i in range(len(vectors)):
            plt.scatter(X[i], Y[i], c=colors[i])
        plt.show()



f1 = "/home/ullas/nltk_trial/corpora/hp/ff_ootp.txt"
f2 = "/home/ullas/nltk_trial/corpora/eragon/eldest.txt"
f3 = "/home/ullas/nltk_trial/corpora/eragon/brisingr.txt"
f4 = "/home/ullas/nltk_trial/corpora/eragon/eragon.txt"

from os import listdir
d = "/home/suhas/PycharmProjects/nlp_pipeline/datasets/bbc/"
#files = [d + '/' +f for f in listdir("/home/ullas/PycharmProjects/nlp_pipeline/datasets/bbc/business")]
import time
st_time = time.time()
lsa = TDMatrix(directory=d, store_to='bbc')
en_time = time.time()
t = en_time - st_time
t /= 60
print("Time taken = " + str(t) + " minutes")
lsa.plot()

#lsa.plot()
#vectors = [lsa.get_doc_vector(i) for i in range(len(files))]
#print("Vectors Calculated")

'''
fn = "/home/ullas/dists.txt"
f = open(fn, 'w+')
for i in range(len(files)):
    for j in range(len(files)):
        if i != j:
            f.write(str(euclidean_similarity(vectors[i], vectors[j])) + '\n')

print("Distances Calculated")

f.close()
'''


