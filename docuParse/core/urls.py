from django.urls import path
from . import views

urlpatterns = [

    path('', views.home, name='home'),
    path('extracted_info/', views.processed, name='processed'),

]