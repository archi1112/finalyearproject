
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import AdminForm, LoginForm
from .models import Admin, User
from .views import *
from django.contrib.auth.decorators import login_required


def admin_signup(request):
    if request.method == 'POST':
        form = AdminForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            email = form.cleaned_data.get('email')
            first_name = form.cleaned_data.get('first_name')
            last_name = form.cleaned_data.get('last_name')
            user = User.objects.create_user(username=username, password=password,
                                            email=email, last_name=last_name, first_name=first_name, user_type=1)
            messages.success(request, 'Account created for {username}!')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('admin_home')
    else:
        form = AdminForm()
        messages.error(request, "Registeration failed")
    return render(request, 'admin_signup.html', {'form': form})


def admin_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None and user.user_type == "1":
                login(request, user)
                messages.success(request, "Logged in Succesfully")
                return redirect('home')
            else:
                messages.error(request, "invalid credentials")
                form.add_error(None, 'Invalid username or password')

    else:
        form = LoginForm()
    errors = []
    for field in form:
        if field.errors:
            errors.append(field.errors)
    print(errors)
    return render(request, 'admin_login.html', {'form': form})


@login_required(login_url='admin_login')
def admin_home(request):
    return render(request, 'admin_home.html')


def admin_logout(request):
    logout(request)
    messages.success(request, "You logged out!")
    return redirect('admin_login')
