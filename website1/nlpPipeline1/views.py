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
from .backend.docsimilarity.similarity import jaccard_similarity, euclidean_similarity
from .backend.docsimilarity.tfidfcalculator import TFIDFVectorizer
from .backend.docsimilarity.tdmgenerator import TDMatrix
import pprint
from sklearn.utils import shuffle


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
        fh1 = open(os.path.join(demo_doc_dir, '00'), 'w+')
        fh2 = open(os.path.join(demo_doc_dir, '01'), 'w+')
        fh1.write(f1.read().decode('utf-8'))
        fh2.write(f2.read().decode('utf-8'))
        fh1.seek(0, 0)
        fh2.seek(0, 0)
        plotdata = demo(fh1, fh2)
        fh1.close()
        fh2.close()
        return render_to_response('nlpPipeline1/twofilesplot.html', {'data': mark_safe(plotdata)})


def compare(request):
    if request.method == 'POST':
        s1 = request.FILES.getlist('source_1')
        s2 = request.FILES.getlist('source_2')
        custom_path = os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/custom')
        s1_dir = os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/custom/source1')
        s2_dir = os.path.join(settings.BASE_DIR, 'nlpPipeline1/data/custom/source2')
        delete_older_posts(s1_dir)
        delete_older_posts(s2_dir)
        table_data = dict()
        # Writing to files
        for f in s1:
            fh = open(os.path.join(s1_dir, f.name), 'w+')
            fh.write(f.read().decode('utf-8'))
            fh.close()

        for f in s2:
            fh = open(os.path.join(s2_dir, f.name), 'w+')
            fh.write(f.read().decode('utf-8'))
            fh.close()

        filenames = [f.name for f in s1]
        rand_files = shuffle(filenames)

        table_data['fnames'] = filenames
        tfidfv = TFIDFVectorizer(custom_path)
        # lsa = TDMatrix(custom_path, store_to='custom')
        # vector_set_1 = [lsa.get_doc_vector(doc_name=os.path.join(s1_dir, file)) for file in filenames]
        # vector_set_2 = [lsa.get_doc_vector(doc_name=os.path.join(s2_dir, file)) for file in filenames]

        # Jaccard Similarity
        for i, file in enumerate(filenames):
            print(file + '\n' + rand_files[i])
            # Initialize list. [jaccard, tfidf, lsa, cosine]
            dis_index = 'dis_' + str(i)
            table_data[file] = list()
            table_data[dis_index] = list()

            # Dissimilarities list : [f1, f2, jaccard, tfidf, lsa, cosine]
            table_data[dis_index].append(file)
            table_data[dis_index].append(rand_files[i])

            # Jaccard Similarity
            # Same Topics
            fh1 = open(os.path.join(s1_dir, file), 'r')
            fh2 = open(os.path.join(s2_dir, file), 'r')
            table_data[file].append(jaccard_similarity(fh1, fh2))
            fh1.close()
            fh2.close()
            # Different Topics
            fh1 = open(os.path.join(s1_dir, file), 'r')
            fh2 = open(os.path.join(s2_dir, rand_files[i]), 'r')
            table_data[dis_index].append(jaccard_similarity(fh1, fh2))
            fh1.close()
            fh2.close()

            # Euclidean with tfidf
            # Same Topics
            table_data[file].append(1 / (1 + tfidfv.distance(os.path.join(s1_dir, file), os.path.join(s2_dir, file))))
            # Different Topics
            table_data[dis_index].append(1 / (1 + tfidfv.distance(os.path.join(s1_dir, file), os.path.join(s2_dir, rand_files[i]))))

            # Euclidean with LSA
            table_data[file].append(4.20)
            table_data[dis_index].append(420)

            # Word2Vec with Cosine
            fh1 = open(os.path.join(s1_dir, file), 'r')
            fh2 = open(os.path.join(s2_dir, file), 'r')
            fh3 = open(os.path.join(s2_dir, rand_files[i]), 'r')
            v1 = im.getDocVector(fh1)
            v2 = im.getDocVector(fh2)
            v3 = im.getDocVector(fh3)
            table_data[file].append(im.getDocSimilarity(v1, v2))
            table_data[dis_index].append(im.getDocSimilarity(v1, v3))
            fh1.close()
            fh2.close()
            fh3.close()

        return render_to_response('nlpPipeline1/tabulate.html', {'data' : mark_safe(table_data)})


def fetch_documents(request):
    n_docs = int(request.GET.get('n_docs')) if request.GET.get('n_docs') != '' else 1000
    reddit = request.GET.get('reddit')
    doc_dir = os.path.join(settings.BASE_DIR, 'reddit/')

    downloadRedditDocs(doc_dir, n_docs, reddit)


    createPickles.generate_pickle_files(doc_dir, 'pickles/')
    complete.run(doc_dir, 'pickles/')
    json_data = json.load(open(os.path.join(settings.BASE_DIR, 'nlpPipeline1/static/nlpPipeline1/js/plot.json')))
    return render_to_response('nlpPipeline1/plot.html', {'data': mark_safe(json_data)})


def fetch_offline_documents(request, file_dir):
    rel_dir = 'nlpPipeline1/data/offlinefiles/' + file_dir
    pickle_path = 'nlpPipeline1/data/offlinefiles/pickles'
    doc_dir = os.path.join(settings.BASE_DIR, rel_dir)
    pickle_dir = os.path.join(settings.BASE_DIR, pickle_path)

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