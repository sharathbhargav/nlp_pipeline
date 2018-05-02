def get_file_name(title):
    name = str()
    for word in title.split(' '):
        name += word.lower() + '_'
    return name.replace('/', '|')[:-1]


file_location = "/home/ullas/PycharmProjects/nlp_pipeline/website/reddit/"