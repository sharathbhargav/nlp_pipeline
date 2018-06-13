import json
from collections import Counter
from django.conf import settings
import os

json_loc = os.path.join(settings.BASE_DIR, 'static/doccer/js')


class PlottingData:
    def __init__(self):
        self._filenames = None
        self.n_files = None
        self._points = None
        self._colors = None
        self._n_clusters = None
        self._named_entities = None
        self._clusters = None
        self._cluster_points = None
        self.__color_ref = ['#E82C0C', '#3AFF00', '#0092FF', '#FFE800', '#FF8000', '#0DE7FF', '#E83C4B']

    def set_filenames(self, fnames):
        self._filenames = fnames
        self.n_files = len(fnames)

    def set_points(self, points_array):
        self._points = dict()
        for i, file in enumerate(self._filenames):
            self._points[file] = points_array[i]

    def set_colors(self, labels):
        self._colors = dict()
        for i, file in enumerate(self._filenames):
            self._colors[file] = self.__color_ref[labels[i]]

    def set_clusters(self, n_clusters, clusters, cluster_points):
        self._n_clusters = n_clusters
        self._clusters = clusters
        self._cluster_points = cluster_points

    def set_named_entities(self, entities_dict):
        self._named_entities = entities_dict

    def prepare_to_plot(self):
        data = dict()
        data['fnames'] = self._filenames
        for file in self._filenames:
            filedata = dict()
            filedata['xy'] = self._points[file].tolist()
            filedata['color'] = self._colors[file]
            filedata['cluster'] = None
            for key, value in self._clusters.items():
                if file in self._clusters[key]:
                    filedata['cluster'] = key
            filedata['org_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['org'])[filedata['cluster']]).most_common(10)
            ]
            filedata['person_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['persons'])[filedata['cluster']]).most_common(10)
            ]
            filedata['place_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['places'])[filedata['cluster']]).most_common(10)
            ]
            filedata['loc_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['loc'])[filedata['cluster']]).most_common(10)
            ]
            filedata['noun_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['nouns'])[filedata['cluster']]).most_common(10)
            ]
            filedata['summary'] = [
                ent[0] for ent in Counter(dict(self._named_entities['summary'])[filedata['cluster']]).most_common(25)
            ]
            data[file] = filedata
        data['n_clusters'] = self._n_clusters
        for i in range(self._n_clusters):
            clusterdata = dict()
            clusterdata['xy'] = self._cluster_points[i].tolist()
            clusterdata['color'] = self.__color_ref[i]
            clusterdata['org_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['org'])[i]).most_common(10)
            ]
            clusterdata['person_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['persons'])[i]).most_common(10)
            ]
            clusterdata['place_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['places'])[i]).most_common(10)
            ]
            clusterdata['loc_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['loc'])[i]).most_common(10)
            ]
            clusterdata['noun_entities'] = [
                ent[0] for ent in Counter(dict(self._named_entities['nouns'])[i]).most_common(10)
            ]
            clusterdata['summary'] = [
                ent[0] for ent in Counter(dict(self._named_entities['summary'])[i]).most_common(25)
            ]
            data[i] = clusterdata
        with open(os.path.join(json_loc, 'plot.json'), 'w+') as f:
            json.dump(data, f, indent=4)

    def status(self):
        print("Number of Files : " + str(self.n_files))
        print("Files : " + str(self._filenames))
        print("Points : \n", self._points)
        print("Colors : \n", self._colors)
        print("Number of CLusters : " + str(self._n_clusters))
        print("Clusters : \n", self._clusters)