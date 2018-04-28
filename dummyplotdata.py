from plotting.plotdata import PlottingData
from website.doccer.redditbot.util import file_location
from os import listdir
from random import uniform, randint
import json
from docprocessing.util import get_files

pd = PlottingData()

# filenames = [file_location + 'hot/' + file for file in listdir(file_location + 'hot')]
filenames = get_files("/home/ullas/PycharmProjects/nlp_pipeline/datasets/bbc/")
pd.set_filenames(filenames)

points = []
labels = []
for i in range(len(filenames)):
    temp_points = [uniform(-1, 1), uniform(-1, 1)]
    points.append(temp_points)
    labels.append(randint(0, 3))

pd.set_points(points)
pd.set_colors(labels)

clusters = dict()
for i in range(4):
    clusters[i] = list()

for i, val in enumerate(labels):
    clusters[val].append(filenames[i])

pd.set_clusters(4, clusters)

pd.prepare_to_plot()
