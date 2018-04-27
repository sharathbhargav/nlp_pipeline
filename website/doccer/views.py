from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from .redditbot import postman as pm
import json
from django.utils.safestring import mark_safe


def index(request):
    return render(request, 'doccer/index.html')


def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    if reddit == 'hot':
        pm.get_hot_posts(n_docs)
    elif reddit == 'new':
        pm.get_new_posts(n_docs)
    elif reddit == 'rising':
        pm.get_rising_posts(n_docs)
    elif reddit == 'controversial':
        pm.get_controversial_posts(n_docs)
    elif reddit == 'top':
        pm.get_top_posts(n_docs)
    elif reddit == 'gilded':
        pm.get_gilded_posts(n_docs)
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
