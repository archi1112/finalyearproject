from django.shortcuts import render, redirect
from .forms  import *
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


faceRecognition=FaceRecognition()


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
# def scan(request):
#     emp_id = faceRecognition.recognizeFace()
#     print(emp_id)
#     data = {'emp_id': emp_id}
#     return JsonResponse(data)

@login_required
def markAttendance(request):
    emp_id = faceRecognition.recognizeFace()
    emp_id2=recognisevoice()
    print(emp_id2)
    print(emp_id)
    try:
        employee = Employee.objects.get(emp_id=emp_id)
        attendance, created = Attendance.objects.get_or_create(
            emp_id=employee, date=date.today())
        if not attendance.status:
            attendance.status = True
            attendance.save()
        return HttpResponse('Attendance marked',emp_id)
    except Employee.DoesNotExist:
        return HttpResponse('Employee does not exist')


def Greeting(request, emp_id):
    emp_id = emp_id
    context = {
        'user': Employee.objects.get(emp_id=emp_id)
    }
    return render(request, 'greeting.html', context=context)

@login_required
def currentEmployeeAttendance(request):
        print("IN")
        print(request)
        if request.user.is_authenticated and request.user.user_type == 2:
            emp_id = request.user.emp_id
            print(emp_id)
            try:
                attendance = Attendance.objects.get(emp_id=emp_id)
                print(attendance)
            except Attendance.DoesNotExist:
                attendance = None
            return render(request, 'currentEmployeeAttendance.html', {'attendance': attendance})
        else:
            return redirect('employee_login')
