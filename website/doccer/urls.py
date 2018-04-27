from django.urls import path


from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('fetch/', views.fetch_documents, name='fetcher'),
    path('displaydoc/<str:file_loc>', views.display_file, name='file_display'),
]
