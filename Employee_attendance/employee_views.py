from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import *
from django.contrib import messages


@login_required
def employee_home(request):
    return render(request, 'employee_home.html')

def employee_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            # user = authenticate(request, username=username, password=password)
            # if user is None:
            #     print('Authentication failed: {}'.format(authenticate(request)))
            # if user is not None and user.user_type==2:
            user = User.objects.filter(username=username).first()
            if user is not None and user.user_type=="2":
                # user exists, now check if password is correct
                if user.check_password(password):
                            login(request, user)
                return redirect('employee_home')
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
    return redirect('employee_login')
