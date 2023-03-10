from django.shortcuts import render, redirect
from .detection import FaceRecognition
# from .attendance_views import AttendanceManager
from .forms import *
from django.contrib import messages
from . employee_views import employee_home
from .admin_views import *

faceRecognition = FaceRecognition()
attendance = Attendance()


def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method=="POST":
        form=EmployeeForm(request.POST or None)
        if form.is_valid():
            emp_id=request.POST['emp_id']
            user=form.save(commit=False)
            user.user_type=2
            user.save()
            Employee.objects.create(user=user, emp_id=request.POST['emp_id'],gender=request.POST['gender'], address=request.POST['address'])
            print(user)
            addFace(emp_id)
            return redirect('home')
        else:
            messages.error(request,"Account registeration failed")
    else:
        form=EmployeeForm()
    return render(request,'register.html',{'form':form})


def addFace(emp_id):
    try:
        print("IN add face",emp_id)
        faceRecognition.faceDetect(emp_id)
        faceRecognition.trainFace()
    except:
        print("Error occured")

