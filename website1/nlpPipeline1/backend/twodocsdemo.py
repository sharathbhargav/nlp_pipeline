from . import individualModules as im
from .processing import cleanup, wordify
import numpy as np
from sklearn.decomposition import PCA


def _get_word_vectors(sentences):
    word_vectors = list()
    word_list = list(set(wordify([sentences])[0]))
    used_words = list()
    temp_word_vectors = list()
    for word in word_list:
        try:
            temp_word_vectors.append(im.getWordVector(word))
            used_words.append(word)
        except:
            continue
    word_vector_array = np.asarray(temp_word_vectors, dtype=np.float32)
    pca = PCA(n_components=2)
    word_vector_array = pca.fit_transform(word_vector_array)
    i = 0
    for i, word in enumerate(used_words):
        word_vectors.append([word, word_vector_array[i].tolist()])
    return [used_words, word_vectors]


def demo(fh1, fh2):
    file_1_sents = cleanup(fh1)
    file_2_sents = cleanup(fh2)
    file_1_word_vectors = _get_word_vectors(file_1_sents)
    file_2_word_vectors = _get_word_vectors(file_2_sents)
    common_words = list(set(file_1_word_vectors[0]).intersection(set(file_2_word_vectors[0])))
    print('commons : ' + str(len(common_words)))
    fh1.seek(0,0)
    fh2.seek(0,0)
    doc_similarity = im.getDocSimilarity(im.getDocVector(fh1), im.getDocVector(fh2))
    plot_data = dict()
    plot_data['fname_1'] = fh1.name
    plot_data['fname_2'] = fh2.name
    plot_data['similarity'] = doc_similarity
    plot_data['f1_points'] = [point for point in file_1_word_vectors[1] if point[0] not in common_words]
    plot_data['f2_points'] = [point for point in file_2_word_vectors[1] if point[0] not in common_words]
    plot_data['common_points'] = [point for point in file_1_word_vectors[1] if point[0] in common_words]
    return plot_data




