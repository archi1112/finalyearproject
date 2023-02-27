from django.shortcuts import render,redirect
from .forms import *
from django.contrib import messages,auth
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login


def admin_signup(request):
    if request.method == 'POST':
        form = AdminForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            messages.success(request, f'Your account has been created ! You are now able to log in')
            form.save()
            return redirect('admin_login')
    else:
        form = AdminForm()
    return render(request, 'admin_signup.html', {'form': form})
  

def admin_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            admin = Admin.objects.get(username=username)
            if admin.password == password:
                request.session['admin_id'] = admin.id
                return redirect('admin_home',username)
            else:
                messages.error(request, 'Invalid email or password')
        except Admin.DoesNotExist:
            messages.error(request, 'Invalid email or password')
    return render(request, 'admin_login.html')

def admin_home(request,username):
    username = username
    context ={
        'user' : Admin.objects.get(username = username)
    }
    return render(request,'admin_home.html',context=context)