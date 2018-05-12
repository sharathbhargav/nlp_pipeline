from django.shortcuts import render, render_to_response
from django.http import HttpResponse
import json
from django.utils.safestring import mark_safe
from django.conf import settings
import os
from .backend.twodocsdemo import demo
from .backend import completePipelineExperiment as complete
from .backend import individualModules as im
from .backend import createPickleFilesForDocs as createPickles
from .backend.redditbot import postman as pm
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


def downloadRedditDocs(doc_dir, n_docs, reddit):
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


def two_files_fun(request):
    if request.method == 'POST':
        f1 = request.FILES['file_1']
        f2 = request.FILES['file_2']
        demo_doc_dir = os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/demodocs')
        for file in os.listdir(demo_doc_dir):
            os.remove(os.path.join(demo_doc_dir, file))
        fh1 = open(os.path.join(demo_doc_dir, f1.name), 'w+')
        fh2 = open(os.path.join(demo_doc_dir, f2.name), 'w+')
        fh1.write(f1.read().decode('utf-8'))
        fh2.write(f2.read().decode('utf-8'))
        fh1.seek(0, 0)
        fh2.seek(0, 0)
        plotdata = demo(fh1, fh2)
        return render_to_response('nlpPipeline1/twofilesplot.html', {'data': mark_safe(plotdata)})


def compare(request):
    if request.method == 'POST':
        f = request.FILES.getlist('source_1')
        return HttpResponse(f[0].read())


def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    doc_dir = os.path.join(settings.BASE_DIR, 'reddit/')

    downloadRedditDocs(doc_dir, n_docs, reddit)


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

    """
    rel_dir = 'nlpPipeline1/data/offlinefiles/' + filedir + '/'
    pickle_dir = 'nlpPipeline1/data/offlinefiles/pickles/' + filedir + '/'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    print("Processing offline docs.")
    if len(os.listdir(os.path.join(settings.BASE_DIR, pickle_dir))) == 0:
        print(">>>>>>>> pickles not present")
        createPickles.generate_pickle_files(doc_dir, pickle_dir)
    complete.run(doc_dir, pickle_dir)
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/static/nlpPipeline1/js/plot.json')))

    """

    print(request.data)
    d=request.GET.get('file', '$$$$')
    print(d)
    #print(json_data)
    #x = temp.run()
    #x=json.dumps(x)
    return Response({"success": True, "content": "json"})

  def post(self, request):
    """
    Return a hardcoded response.
    """
    """
    print(type(request.data))
    queryData=request.data

    file1=queryData['file1']
    file2=queryData['file2']
    #x = temp.run()
    #x=json.dumps(x)
    print(file1)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(file2)
    
    """
    filedir='source2'
    rel_dir = 'nlpPipeline1/data/offlinefiles/' + filedir + '/'
    pickle_dir = 'nlpPipeline1/data/offlinefiles/pickles/' + filedir + '/'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    print("Processing offline docs.")
    if len(os.listdir(os.path.join(settings.BASE_DIR, pickle_dir))) == 0:
        print(">>>>>>>> pickles not present")
        createPickles.generate_pickle_files(doc_dir, pickle_dir)
    fileDict=complete.runForAPI(doc_dir, pickle_dir)



    #print(JSONDICT)

    return Response({"content": fileDict})




class fileDetails(APIView):
    def post(self,request):
        queryData = request.data

        file1 = queryData['file1']
        file2 = queryData['file2']
        print("File1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(file1)
        print("file2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(file2)
        demo_doc_dir = os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/apiDocs')
        for file in os.listdir(demo_doc_dir):
            os.remove(os.path.join(demo_doc_dir, file))
        fh1 = open(os.path.join(demo_doc_dir, "file1"), 'w+')
        fh2 = open(os.path.join(demo_doc_dir, "file2"), 'w+')
        fh1.write(file1)
        fh2.write(file2)
        fh1.seek(0, 0)
        fh2.seek(0, 0)
        plotdata = demo(fh1, fh2)


        return Response({'data':plotdata['similarity']})