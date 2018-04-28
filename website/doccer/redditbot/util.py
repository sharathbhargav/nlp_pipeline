def get_file_name(title):
    name = str()
    for word in title.split(' '):
        name += word.lower() + '_'
    name.replace('/', '|')
    return name[:-1]


file_location = "/home/ullas/PycharmProjects/nlp_pipeline/datasets/reddit/"