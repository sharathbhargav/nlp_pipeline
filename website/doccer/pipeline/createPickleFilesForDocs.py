import pickle
import os
from . import individualModules as im
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def generate_pickle_files(fpath):
    print("Generating Pickle files.")
    trainingModelGoogle = KeyedVectors.load_word2vec_format(
        "/home/ullas/PycharmProjects/nlp_pipeline/models/GoogleNews-vectors-negative300.bin",
        binary=True,
        limit=10000)
    im.setModel(trainingModelGoogle)
    file_names = list()
    file_count = 0
    file_handles = list()
    for fname in os.listdir(fpath):
        file = open(os.path.join(fpath, fname), 'r')
        file_handles.append(file)
        file_names.append(fname)
        file_count += 1
    plot_data = im.getPlotValuesOfDocuments(file_handles)
    for fh in file_handles:
        fh.close()
    total1 = np.array(plot_data)
    custom2Pickle = open("/home/ullas/PycharmProjects/nlp_pipeline/website/reddit/pickles/plotValuesOfDocs", "wb")
    pickle.dump(total1, custom2Pickle)
    custom2Pickle.close()
    custom2Names = open("/home/ullas/PycharmProjects/nlp_pipeline/website/reddit/pickles/plotNamesOfDocs", "wb")
    pickle.dump(file_names, custom2Names)
    custom2Names.close()
    print("Done generating pickle files.")