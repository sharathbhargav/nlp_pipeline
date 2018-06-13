from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from .redditbot import postman as pm
import json
from django.utils.safestring import mark_safe
from .pipeline.createPickleFilesForDocs import generate_pickle_files
from .pipeline.completePipelineExperiment import run
from django.conf import settings
import os


def index(request):
    return render(request, 'doccer/index.html')


def delete_older_posts(doc_dir):
    print('Deleting older downloaded posts')
    for file in os.listdir(doc_dir):
        print(str(os.path.join(doc_dir, file)) + ' deleted')
        os.remove(os.path.join(doc_dir, file))
    print('Done deleting older downloaded posts.')


def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    doc_dir = os.path.join(settings.BASE_DIR, 'reddit/')
    print("Fetching reddit posts")
    if reddit == 'Hot':
        doc_dir = os.path.join(doc_dir, 'hot/')
        delete_older_posts(doc_dir)
        pm.get_hot_posts(n_docs)
    elif reddit == 'New':
        doc_dir = os.path.join(doc_dir, 'files/')
        delete_older_posts(doc_dir)
        pm.get_new_posts(n_docs)
    elif reddit == 'Rising':
        doc_dir = os.path.join(doc_dir, 'rising/')
        delete_older_posts(doc_dir)
        pm.get_rising_posts(n_docs)
    elif reddit == 'Controversial':
        doc_dir = os.path.join(doc_dir, 'controversial/')
        delete_older_posts(doc_dir)
        pm.get_controversial_posts(n_docs)
    elif reddit == 'Top':
        doc_dir = os.path.join(doc_dir, 'top/')
        delete_older_posts(doc_dir)
        pm.get_top_posts(n_docs)
    elif reddit == 'Gilded':
        doc_dir = os.path.join(doc_dir, 'gilded/')
        delete_older_posts(doc_dir)
        pm.get_gilded_posts(n_docs)
    print("Done fetching posts.")
    generate_pickle_files(doc_dir, 'pickles/')
    run(doc_dir, 'pickles/')
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'static/doccer/js/plot.json')))
    return render_to_response('doccer/plot.html', {'data': mark_safe(json_data)})


def fetch_offline_documents(request, filedir):
    rel_dir = 'offlinefiles/' + filedir + '/'
    picle_dir = 'offlinefiles/pickles/' + filedir + '/'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    print("Processing offline docs.")
    if len(os.listdir(os.path.join(settings.BASE_DIR, picle_dir))) == 0:
        generate_pickle_files(doc_dir, picle_dir)
    run(doc_dir, picle_dir)
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'static/doccer/js/plot.json')))
    return render_to_response('doccer/plot.html', {'data': mark_safe(json_data)})


def display_file(request, file_loc):
    file_loc = file_loc.replace('+', '/')
    f = open(file_loc, 'r')
    file_data = str()
    next = f.readline()
    while next != "":
        file_data += next + '<br />'
        next = f.readline()
    f.close()
    return HttpResponse(file_data)
