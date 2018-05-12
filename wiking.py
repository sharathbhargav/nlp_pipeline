import wikipedia
import os

list_file = '/home/ullas/movie_list.txt'
fh = open(list_file, 'r')
movies = [m.rstrip('\n') for m in fh.readlines()]


def fileify(s):
    s = s.replace('\n', '')
    s = s.split(' ')
    name = str()
    for c in s:
        name = name + c.lower() + '_'
    return name[:-1]


def get_movie_page(query, search_result):
    for result in search_result:
        if '(film)' in result and query.rstrip() in result:
            return wikipedia.page(result)
    return wikipedia.page(search_result[0])


def get_movie_plot(page_content):
    index = page_content.find('== Plot ==')
    if index == -1:
        return None
    else:
        index += 11
    plot = str()
    for i in range(index, len(page_content)):
        if page_content[i] + page_content[i+1] == '==':
            return plot
        else:
            plot += page_content[i]


d = '/home/ullas/movie_plots'


for m in movies:
    print('Done: ' + m)
    page = get_movie_page(m, wikipedia.search(m))
    fname = os.path.join(d, fileify(m))
    fh = open(fname, 'w+')
    plot = get_movie_plot(page.content)
    if plot == None:
        print('\r--> Unable to get plot for : ' + m)
        fh.close()
        os.remove(fname)
        continue
    fh.write(get_movie_plot(page.content))
    fh.close()

