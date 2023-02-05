from django.contrib import admin
from django.urls import path
from .views import *
from .admin_views import *
from .employee_views import *
from .attendance_views import *
urlpatterns = [
    path('',home,name = "home"),
    path('login/',login,name = 'login'),
    path('greeting/<emp_id>/',Greeting,name='greeting'),
    path('register/',register,name='register'),

    # employee views
    path('employee_home/',employee_home,name='employee_home'),
    path('employee_login/',employee_login,name='employee_login'),


    # admin views
    path('admin_signup/',admin_signup,name='admin_login'),
    path('admin_login/',admin_login,name='admin_login'),
    path('admin_home/',admin_home,name='admin_home'),

    # attendance views
    path('employees_attendance/',employees_attendance,name='employees_attendance')

]
