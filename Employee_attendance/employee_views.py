from django.shortcuts import render,redirect
from .forms import *
from django.contrib import messages

def employee_home(request):
    return render(request,'employee_home.html')

def employee_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            employee = Employee.objects.get(username=username)
            if employee.password == password:
                # request.session['employee_id'] = employee.id
                return redirect('employee_home')
            else:
                messages.error(request, 'Invalid email or password')
        except Employee.DoesNotExist:
            messages.error(request, 'Invalid email or password')
    return render(request, 'employee_login.html')

