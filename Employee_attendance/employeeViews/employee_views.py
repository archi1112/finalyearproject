from django.shortcuts import render,redirect

def employeeHome(request):
    return render(request,'employeeHome.html')
