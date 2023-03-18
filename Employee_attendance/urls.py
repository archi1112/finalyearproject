from django.contrib import admin
from django.urls import path
from .views import *
from .admin_views import *
from .employee_views import *
from .attendance_views import *
from Employee_attendance.attendance_views import Attendance
urlpatterns = [
    path('', home, name="home"),
    path('register/', register, name='register'),
    path('markAttendance/', markAttendance, name='markAttendance'),
    path('greeting/<emp_id>/', Greeting, name='greeting'),

    path('employee_login/', employee_login, name='employee_login'),
    path('employee_home/', employee_home, name='employee_home'),
    path('employee_logout/', employee_logout, name='employee_logout'),
    # admin views
    path('admin_signup/', admin_signup, name='admin_login'),
    path('admin_login/', admin_login, name='admin_login'),
    path('admin_home/', admin_home, name='admin_home'),
    path('admin_logout/', admin_logout, name='admin_logout'),
    path('attendancebydate/',
         attendance_by_date, name='attendance_by_date'),
    path('currentEmployeeAttendance', currentEmployeeAttendance,
         name='currentEmployeeAttendance'),

    path('recordvoice/<emp_id>', recordvoice, name='recordvoice')

]
