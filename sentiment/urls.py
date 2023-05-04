from django.urls import re_path, include
from . import views
from django.http import HttpResponse


app_name = 'sentiment'


urlpatterns = [
    re_path(r'^$', views.sentiment_analysis, name="sentiment_anaylsis"),
    re_path(r'^type/$', views.sentiment_analysis_type, name="sentiment_analysis_type"),
    re_path(r'^import/$', views.sentiment_analysis_import, name="sentiment_analysis_import"),
]