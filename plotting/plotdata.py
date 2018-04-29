import json

json_loc = "/home/ullas/PycharmProjects/nlp_pipeline/website/doccer/static/doccer/js/"


class PlottingData:
    def __init__(self):
        self._filenames = None
        self.n_files = None
        self._points = None
        self._colors = None
        self._n_clusters = None
        self._named_entities = None
        self._clusters = None

    def set_filenames(self, fnames):
        self._filenames = fnames
        self.n_files = len(fnames)

    def set_points(self, points_array):
        self._points = dict()
        for i, file in enumerate(self._filenames):
            self._points[file] = points_array[i]

    def set_colors(self, labels):
        self._colors = dict()
        color_ref = ['#E82C0C', '#3AFF00', '#0092FF', '#FFE800', '#FF8000', '#0DE7FF', '#E83C4B']
        for i, file in enumerate(self._filenames):
            self._colors[file] = color_ref[labels[i]]

    def set_clusters(self, n_clusters, clusters):
        self._n_clusters = n_clusters
        self._clusters = clusters

    def prepare_to_plot(self):
        data = dict()
        data['fnames'] = self._filenames
        for file in self._filenames:
            filedata = dict()
            filedata['xy'] = self._points[file]
            filedata['color'] = self._colors[file]
            filedata['cluster'] = None
            for key, value in self._clusters.items():
                if file in self._clusters[key]:
                    filedata['cluster'] = key
            data[file] = filedata
        with open(json_loc + 'plot.json', 'w+') as f:
            json.dump(data, f, indent=4)

    def status(self):
        print("Number of Files : " + str(self.n_files))
        print("Files : " + str(self._filenames))
        print("Points : \n", self._points)
        print("Colors : \n", self._colors)
        print("Number of CLusters : " + str(self._n_clusters))
        print("Clusters : \n", self._clusters)