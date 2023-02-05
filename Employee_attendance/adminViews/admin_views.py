from django.shortcuts import render,redirect
from .forms import *
from django.contrib import messages,auth

def admin_signup(request):
    if request.method==POST:
        username=request.POST['username']
        
    return render(request,'admin_signup.html')

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'admin' and password == '123':
            return redirect('admin_home')
            print('done')
        else:
            print('invalid')
    return render(request, 'admin_login.html')

def admin_home(request):
    return render(request,'admin_home.html')