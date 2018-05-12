import json


def fileify(s):
    s = s.replace('\n', '')
    s = s.split(' ')
    name = str()
    for c in s:
        name = name + c.lower() + '_'
    return name[:-1]


imdb_dir = 'movies/imdb/'
wiki_dir = 'movies/wiki/'

genres = open('genres.txt', 'r')
titles = open('titles.txt', 'r')
imdb = open('imdb.txt', 'r')
wiki = open('wiki.txt', 'r')
w = open('wiki1.txt', 'w+')

imdb_plots = list()
wiki_plots = list()
titles_list = list()
genres_list = list()


for plot in imdb.read().split('BREAKS HERE'):
    if plot == '':
        continue
    imdb_plots.append(plot.replace('\n', ''))

for plot in wiki.read().split('BREAKS HERE'):
    if plot == '':
        continue
    wiki_plots.append(plot.replace('\n', ''))

for t in titles.readlines():
    titles_list.append(t)

for line in genres.readlines():
    if len(line[1:-1].split(',')) > 1:
        genres_list.append(line[1:-1].split(',')[0][3:-1])
    elif len(line[1:-1].split(',')) == 1:
        genres_list.append(line[1:-1].split(',')[0][3:-2])

'''
for i, plot in enumerate(imdb_plots):
    fname = imdb_dir + fileify(titles_list[i])
    f = open(fname, 'w+')
    f.write(plot)
    f.close()

for i, plot in enumerate(wiki_plots):
    fname = wiki_dir + fileify(titles_list[i])
    f = open(fname, 'w+')
    f.write(plot)
    f.close()
'''

genres_set = list(set(genres_list))

movie_clusters = dict()

for i, tit in enumerate(titles_list):
    x = genres_set.index(genres_list[i])
    movie_clusters[fileify(tit)] = x

with open('movies.json', 'w+') as f:
    json.dump(movie_clusters, f, indent=4)
