from django.shortcuts import render,redirect
from Employee_attendance.detection import FaceRecognition
from .forms import *
from django.contrib import messages,auth
from django.views.decorators.csrf import csrf_protect
from .admin_views import *


faceRecognition = FaceRecognition()

def home(request):
    return render(request,'home.html')


def register(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST or None)
        if form.is_valid():
            form.save()
            print("IN HERE")
            messages.success(request,"SuceessFully registered")
            addFace(request.POST['emp_id'])
            redirect('home')
        else:
            messages.error(request,"Account registered failed")
    else:
        form = EmployeeForm()

    return render(request, 'employee_login.html', {'form':form})

def addFace(emp_id):
    emp_id = emp_id
    faceRecognition.faceDetect(emp_id)
    faceRecognition.trainFace()
    return redirect('/')

def login(request):
    emp_id = faceRecognition.recognizeFace()
    faceRecognition.faceDetect(emp_id)
    print(emp_id)
    return redirect('greeting' ,str(emp_id))

def Greeting(request,emp_id):
    emp_id = int(emp_id)
    context ={
        'user' : Employee.objects.get(emp_id = emp_id)
    }
    return render(request,'greeting.html',context=context)

