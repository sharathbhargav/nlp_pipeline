from docprocessing.processing import cleanup
from docprocessing.processing import fusion
#from docsimilarity.tdmgenerator import TDMatrix
from math import pow
from math import sqrt

# Takes two file handlers, returns the Jaccard Similarity between them.
# 0.0 means dissimilar and 1.0 means identical.
def jaccard_similarity(fh_one, fh_two):
    doc_one = cleanup(fh_one)
    doc_two = cleanup(fh_two)
    docs = fusion([doc_one, doc_two])
    set_doc_one = set(docs[0].split(' '))
    if '' in set_doc_one: set_doc_one.remove('')
    set_doc_two = set(docs[1].split(' '))
    if '' in set_doc_two: set_doc_two.remove('')
    js = len(set_doc_one.intersection(set_doc_two)) / len(set_doc_one.union(set_doc_two))
    return js


def euclidean_similarity(d_vec_one, d_vec_two):
    _sq_sum = 0
    for i in range(len(d_vec_one)):
        _sq_sum += pow((d_vec_two[i] - d_vec_one[i]), 2)
    e_dist = sqrt(_sq_sum)
    return e_dist

'''
f1 = "/home/ullas/nltk_trial/corpora/hp/ff_ootp.txt"
f2 = "/home/ullas/nltk_trial/corpora/eragon/eldest.txt"
f3 = "/home/ullas/nltk_trial/corpora/eragon/brisingr.txt"
f4 = "/home/ullas/nltk_trial/corpora/eragon/eragon.txt"

from os import listdir
import time
import pickle
import datetime


d = "/home/ullas/PycharmProjects/nlp_pipeline/datasets/bbc/business"
files = [d + '/' +f for f in listdir("/home/ullas/PycharmProjects/nlp_pipeline/datasets/bbc/business")]

lsa = TDMatrix([f1, f2, f3, f4])

#f = open('../models/tdm_model_2018-03-22_bbc_business', 'rb')
#lsa = pickle.load(f)

v1 = lsa.get_doc_vector(0)
v2 = lsa.get_doc_vector(1)
v3 = lsa.get_doc_vector(2)
v4 = lsa.get_doc_vector(3)

#dname = "../models/"
#fname = dname + "tdm_model_" + str(datetime.date.today()) + '_' + 'bbc_business'
#fObj = open(fname, 'wb')
#pickle.dump(lsa, fObj)
#fObj.close()


print(len(v1))


print(euclidean_similarity(v1, v2))
print(euclidean_similarity(v1, v3))
print(euclidean_similarity(v1, v4))

print(euclidean_similarity(v2, v3))
print(euclidean_similarity(v2, v4))

print(euclidean_similarity(v3, v4))
'''