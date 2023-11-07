from django.urls import path
from . import views

# Create your views here.


app_name = 'poses'

urlpatterns = [
    path('index/', views.index, name="index"),
]
