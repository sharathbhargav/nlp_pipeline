from django.shortcuts import render, render_to_response
from django.http import HttpResponse
import json
from django.utils.safestring import mark_safe
from django.conf import settings
import os

import nlpPipeline1.backend
from nlpPipeline1.backend import completePipelineExperiment as complete
from nlpPipeline1.backend import individualModules as im
from nlpPipeline1.backend import createPickleFilesForDocs as createPickles
from nlpPipeline1.backend.redditbot import postman as pm
from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
# Create your views here.
def index1(request):

    return render(request,'nlpPipeline1/index1.html',{})

def dataDump(request):
    im.printStopWords()
    x=complete.run()
    return render(request, 'nlpPipeline1/dataDump.html', {'data':x})




def index(request):
    return render(request, 'nlpPipeline1/index.html')


def delete_older_posts(doc_dir):
    print('Deleting older downloaded posts')
    for file in os.listdir(doc_dir):
        print(str(os.path.join(doc_dir, file)) + ' deleted')
        os.remove(os.path.join(doc_dir, file))
    print('Done deleting older downloaded posts.')


def downloadRedditDocs(doc_dir,n_docs,reddit):
    print("Fetching reddit posts")
    if reddit == 'Hot':
        doc_dir = os.path.join(doc_dir, 'hot/')
        delete_older_posts(doc_dir)
        pm.get_hot_posts(n_docs)
    elif reddit == 'New':
        doc_dir = os.path.join(doc_dir, 'new/')
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





def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    doc_dir = os.path.join(settings.BASE_DIR, 'reddit/')

    downloadRedditDocs(doc_dir,n_docs,reddit)


    createPickles.generate_pickle_files(doc_dir, 'pickles/')
    complete.run(doc_dir, 'pickles/')
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/static/nlpPipeline1/js/plot.json')))
    return render_to_response('nlpPipeline1/plot.html', {'data': mark_safe(json_data)})


def fetch_offline_documents(request, filedir):
    rel_dir = 'nlpPipeline1/data/offlinefiles/' + filedir + '/'
    pickle_dir = 'nlpPipeline1/data/offlinefiles/pickles/' + filedir + '/'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    print("Processing offline docs.")
    if len(os.listdir(os.path.join(settings.BASE_DIR, pickle_dir))) == 0:
        print(">>>>>>>> pickles not present")
        createPickles.generate_pickle_files(doc_dir, pickle_dir)
    complete.run(doc_dir, pickle_dir)
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/static/nlpPipeline1/js/plot.json')))
    return render_to_response('nlpPipeline1/plot.html', {'data': mark_safe(json_data)})


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



def get1(request):
    """
    Return a hardcoded response.
    """
    print(request)
    d=request.GET.get('param1', '$$$$')
    print(d)
    #x = temp.run()
    #x=json.dumps(x)
    return Response({"success": True, "content": "hello gube"})





class DemoAPI(APIView):
  """
  A custom endpoint for GET request.
  """
  def get(self, request):
    """
    Return a hardcoded response.
    """
    filedir=request.GET.get("filePath",'new')
    rel_dir = 'nlpPipeline1/data/offlinefiles/' + filedir + '/'
    pickle_dir = 'nlpPipeline1/data/offlinefiles/pickles/' + filedir + '/'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    print("Processing offline docs.")
    if len(os.listdir(os.path.join(settings.BASE_DIR, pickle_dir))) == 0:
        print(">>>>>>>> pickles not present")
        createPickles.generate_pickle_files(doc_dir, pickle_dir)
    complete.run(doc_dir, pickle_dir)
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/static/nlpPipeline1/js/plot.json')))



    print(request)
    d=request.GET.get('param2', '$$$$')
    print(d)
    print(json_data)
    #x = temp.run()
    #x=json.dumps(x)
    return Response({"success": True, "content": json_data})

  def post(self, request):
    """
    Return a hardcoded response.
    """
    print(request)
    d=request.GET.get('param1', '$$$$')
    print(d)
    #x = temp.run()
    #x=json.dumps(x)
    return Response({"success": True, "content": "hello post1"})



class testAPI(APIView):
    def get(self,request):
        filePath=request.GET.get('filePath1','')
        param1=request.GET.get('param1','')

        print(param1)
        print(filePath)

        return Response({"success":True,"content":"Hello dumma"})