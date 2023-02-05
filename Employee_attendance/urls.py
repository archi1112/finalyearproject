from django.contrib import admin
from django.urls import path
from .views import *
from adminViews.admin_views.py import *
from employeeViews.employee_views import *
urlpatterns = [
    path('',home,name = "home"),
    path('register/',register,name = 'register'),
    path('login/',login,name = 'login'),
    path('greeting/<face_id>/',Greeting,name='greeting'),

    # employee views
    path('employee_home/',employee_home,name='employee_home')
    path('employee_login/',employee_login,name='employee_login')


    # admin views
    path('admin_signup/',admin_login,name='admin_login'),
    path('admin_login/',admin_login,name='admin_login'),
    path('admin_home/',admin_home,name='admin_home'),

]
