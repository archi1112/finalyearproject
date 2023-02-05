from django.shortcuts import render,redirect

def employee_home(request):
    return render(request,'employee_home.html')

def employee_login(request):
    return render(request,'employee_login.html')
