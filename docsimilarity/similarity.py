from docprocessing.processing import cleanup
from docprocessing.processing import fusion
from docsimilarity.tdmgenerator import TDMatrix
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


def euclidean_similarity(f1, f2):
    lsa = TDMatrix([f1, f2])
    d_vec_one = lsa.get_doc_vector(0)
    d_vec_two = lsa.get_doc_vector(1)
    _sq_sum = 0
    for i in range(len(d_vec_one)):
        _sq_sum += pow((d_vec_two[i] - d_vec_one[i]), 2)
    e_dist = sqrt(_sq_sum)
    return e_dist

f1 = "/home/ullas/nltk_trial/corpora/hp/ff_ootp.txt"
f2 = "/home/ullas/nltk_trial/corpora/eragon/eldest.txt"
f3 = "/home/ullas/nltk_trial/corpora/eragon/brisingr.txt"
f4 = "/home/ullas/nltk_trial/corpora/eragon/eragon.txt"


print(euclidean_similarity(f1, f2))
print(euclidean_similarity(f1, f3))
print(euclidean_similarity(f1, f4))

print(euclidean_similarity(f2, f3))
print(euclidean_similarity(f2, f4))

print(euclidean_similarity(f3, f4))
