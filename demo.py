from docsimilarity.similarity import jaccard_similarity, euclidean_similarity
from os import listdir
from docsimilarity.tdmgenerator import TDMatrix
from docsimilarity.tfidfcalculator import TFIDFVectorizer

custom_data_set_path = "/home/ullas/project/nlp_pipeline/datasets/custom/"
file_names = [file.split('.')[0] for file in listdir(custom_data_set_path + 'source1')]
source1 = [custom_data_set_path + 'source1/' + file + '.txt' for file in file_names]
source2 = [custom_data_set_path + 'source2/' + file + '.txt' for file in file_names]
n = len(file_names)
#print(source1)
#print(source2)

# Demo for Jaccard Similarity Index

print("\nJaccard Similarity")
s = 0
for file in file_names:
    f_name_1 = custom_data_set_path + 'source1/' + file + '.txt'
    f_name_2 = custom_data_set_path + 'source2/' + file + '.txt'
    fh1 = open(f_name_1, 'r')
    fh2 = open(f_name_2, 'r')
    print(file + ' : ' + str(jaccard_similarity(fh1, fh2)))
    #s += jaccard_similarity(fh1, fh2)
    fh1.close()
    fh2.close()
#print(s / 10)

# Demo for Euclidean with LSA

lsa = TDMatrix(directory=custom_data_set_path, store_to='custom')
vector_set_1 = [lsa.get_doc_vector(doc_name=file) for file in source1]
vector_set_2 = [lsa.get_doc_vector(doc_name=file) for file in source2]
print("\nEuclidean Similarity with LSA")
s = 0
for i in range(n):
    print(file_names[i] + ' : ' + str(1 / (1 + euclidean_similarity(vector_set_1[i], vector_set_2[i]))))
    #s += (1 / (1 + euclidean_similarity(vector_set_1[i], vector_set_2[i])))
#print(s / 10)

# Demo for Euclidean with TF-IDF

tfidfv = TFIDFVectorizer(custom_data_set_path)
print("\nEuclidean Similarity with TF-IDF")
s = 0
for i in range(n):
    print(file_names[i] + ' : ' + str(1 / (1 + tfidfv.distance(source1[i], source2[i]))))
    #s += (1 / (1 + tfidfv.distance(source1[i], source2[i])))
#print(s / 10)


