from django.conf import settings
import os


def get_file_name(title):
    name = str()
    for word in title.split(' '):
        name += word.lower() + '_'
    return name.replace('/', '|')[:-1]


file_location = os.path.join(settings.BASE_DIR, 'reddit/')
