from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from .redditbot import postman as pm
import json
from django.utils.safestring import mark_safe
from .pipeline.createPickleFilesForDocs import generate_pickle_files
from .pipeline.completePipelineExperiment import run


def index(request):
    return render(request, 'doccer/index.html')


def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    doc_dir = "/home/ullas/PycharmProjects/nlp_pipeline/website/reddit/"
    print("Fetching reddit posts")
    if reddit == 'Hot':
        pm.get_hot_posts(n_docs)
        doc_dir += 'hot/'
    elif reddit == 'New':
        pm.get_new_posts(n_docs)
        doc_dir += 'new/'
    elif reddit == 'Rising':
        pm.get_rising_posts(n_docs)
        doc_dir += 'rising/'
    elif reddit == 'Controversial':
        pm.get_controversial_posts(n_docs)
        doc_dir += 'controversial/'
    elif reddit == 'Top':
        pm.get_top_posts(n_docs)
        doc_dir += 'top/'
    elif reddit == 'Gilded':
        pm.get_gilded_posts(n_docs)
        doc_dir += 'gilded/'
    print("Done fetching posts.")
    generate_pickle_files(doc_dir)
    run(doc_dir)
    json_data = json.load(open("/home/ullas/PycharmProjects/nlp_pipeline/website/static/doccer/js/plot.json"))
    return render_to_response('doccer/plot.html', {'data' : mark_safe(json_data)})


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
