from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    path('index', views.index1, name='index'),
    path('dump',views.dataDump,name='dump'),
    url(r'api/demo',views.DemoAPI.as_view()),
    url(r'api/get',views.testAPI.as_view()),



    path('', views.index, name='index'),
    path('displaydoc/<str:file_loc>', views.display_file, name='file_display'),

    path('tabulate/', views.compare, name='tabulate'),
    path('docdemo/', views.two_files_fun, name='doc_demo'),
    path('fetch/', views.fetch_documents, name='fetcher'),
    path('offline/<str:file_dir>', views.fetch_offline_documents, name='offline_fetcher'),
]

urlpatterns = format_suffix_patterns(urlpatterns)