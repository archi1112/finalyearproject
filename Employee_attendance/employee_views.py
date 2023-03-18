from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from  .views import *
from django.contrib import messages
# from .views import home


@login_required(login_url='employee_login')
def employee_home(request):
    return render(request, 'employee_home.html')

def employee_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            print(username,password)
            user = User.objects.filter(username=username).first()
            print("user=",user)
            if user is not None and user.user_type=="2":
                # if user.check_password(password):
                if user.username==username and user.password==password:
                    print("Logging in")
                    login(request, user)
                    return redirect('home')
            else:
                form.add_error(None, 'Invalid username or password')
            
    else:
        form = LoginForm()
    errors = []
    for field in form:
        if field.errors:
            errors.append(field.errors)
    print(errors)
    return render(request, 'employee_login.html', {'form': form})


def employee_logout(request):
    logout(request)
    return redirect('home')
