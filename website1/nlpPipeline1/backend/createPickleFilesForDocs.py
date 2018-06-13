import pickle
import os
from . import individualModules as im
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from django.conf import settings


def generate_pickle_files(fpath, pdir):
    print("Generating Pickle files.")
    """
    trainingModelGoogle = KeyedVectors.load_word2vec_format(
        os.path.join(settings.BASE_DIR, 'backend/models/GoogleNews-vectors-negative300.bin'),
        binary=True,
        limit=10000)
    im.setModel(trainingModelGoogle)
    """
    file_names = []
    file_count = 0
    file_handles = []
    for fname in os.listdir(fpath):
        file = open(os.path.join(fpath, fname), 'r')
        file_handles.append(file)
        file_names.append(fname)
        file_count += 1
    plot_data = im.getPlotValuesOfDocuments(file_handles)
    for fh in file_handles:
        fh.close()
    total1 = np.array(plot_data)
    print(pdir)
    custom2Pickle = open(os.path.join(pdir, 'plotValuesOfDocs'), "wb")
    pickle.dump(total1, custom2Pickle)
    custom2Pickle.close()
    custom2Names = open(os.path.join(pdir, 'plotNamesOfDocs'), "wb")
    pickle.dump(file_names, custom2Names)
    custom2Names.close()
    print("Done generating pickle files.")