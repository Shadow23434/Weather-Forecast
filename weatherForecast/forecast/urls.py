from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_view, name='home'),
    path('api/capital/historical/', views.capital_historical, name='capital_historical_api'),
    path('capitals/', views.capitals_list, name='capitals_list'),
    path('capitals/fetch/', views.manual_historical_data, name='manual_historical_data'),
]