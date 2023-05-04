from django.urls import re_path, include
from . import views

app_name = 'sentiment_home'

urlpatterns = [
    re_path(r'^$', views.choose_sentiment_or_emotion, name="choose_sentiment_or_emotion"),
]