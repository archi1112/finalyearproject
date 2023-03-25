from django.shortcuts import render, redirect
from .forms import *
from .views import *
from django.contrib import messages
from datetime import date
from django.contrib.auth.models import User
from .models import Attendance
from .employee_views import employee_home
from django.contrib.auth.decorators import login_required
from .detection import *
from django.http import JsonResponse
from django.http import HttpResponse


faceRecognition = FaceRecognition()

@login_required
def currentEmployeeAttendance(request):
    # get the currently logged in employee
    employee = Employee.objects.get(user=request.user)
    if request.method == 'POST':
        start_date = request.POST['start_date']
        end_date = request.POST['end_date']
        # get the attendance records for the employee
        attendance = Attendance.objects.filter(emp_id=employee, date__range=(start_date, end_date))
    else:
        attendance = Attendance.objects.filter(emp_id=employee)
    
    # pass the attendance records and employee to the template
    context = {
        'attendance_data': attendance,
        'employee': employee
    }
    
    # render the template with the context
    return render(request, 'currentEmployeeAttendance.html', context)

# changes require
def attendance_view(request):
    if request.method == 'POST':
        employee_id = request.POST['employee_id']
        start_date = request.POST['start_date']
        end_date = request.POST['end_date']
        attendances = Attendance.objects.filter(emp_id=employee_id, date__range=(start_date, end_date))
        
    else:
        attendances = Attendance.objects.all()
    
    context = {'attendances': attendances}
    
    return render(request, 'admin_view.html', context)

def attendance_by_date(request):
    if request.method == 'POST':
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data['date']
            attendance = Attendance.objects.filter(date=date)
            context = {
                'attendance': attendance,
                'date': date,
            }
            return render(request, 'attendanceDetails.html', context)
        else:
            form = DateForm()

        context = {
            'form': form,
        }
        return render(request, 'admin_home.html', context)

def markAttendance(request):
    emp_id = faceRecognition.recognizeFace()
    if emp_id == None:
        return HttpResponse('Employee does not exist')

    return render(request, 'recognisevoice.html')


def process_voice_recognition(request):
    emp_id = recognisevoice(request)
    try:
        employee = Employee.objects.get(emp_id=emp_id)
        attendance, created = Attendance.objects.get_or_create(
            emp_id=employee, date=date.today())
        if not attendance.status:
            attendance.status = True
            attendance.save()
        return HttpResponse('Attendance marked for employee ID: {}'.format(emp_id))
    except Employee.DoesNotExist:
        return HttpResponse('Employee does not exist')


def Greeting(request, emp_id):
    emp_id = emp_id
    context = {
        'user': Employee.objects.get(emp_id=emp_id)
    }
    return render(request, 'greeting.html', context=context)



