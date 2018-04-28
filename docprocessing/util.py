from os.path import isdir
from os import listdir


def get_files(dirname):
    if dirname[-1] == '/':
        pass
    else:
        dirname += '/'
    list_of_files = list()
    for file in listdir(dirname):
        file = dirname + file
        if isdir(file):
            list_of_files.extend(get_files(file))
        else:
            list_of_files.append(file)
    return list_of_files