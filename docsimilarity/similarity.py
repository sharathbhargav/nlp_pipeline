from docprocessing.processing import cleanup
from docprocessing.processing import fusion


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