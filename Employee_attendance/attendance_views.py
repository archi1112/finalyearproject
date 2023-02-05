from django.shortcuts import render,redirect
from .forms import *
from django.contrib import messages,auth
from django.contrib.auth.models import User

def employees_attendance(request):
    return render(request,'employees_attendance.html')