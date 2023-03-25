from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import *
from django.contrib.auth import get_user_model

class EmployeeForm( forms.ModelForm):
    emp_id=forms.IntegerField(label="EMployee ID",widget=forms.TextInput(attrs={"class": "form-control"}))
    address = forms.CharField(label="Address", max_length=50, widget=forms.TextInput(attrs={"class": "form-control"}))
    gender_choice = (("Male", "Male"), ("Female", "Female"))
    gender = forms.ChoiceField(label="gender", choices=gender_choice, widget=forms.Select(attrs={"class": "form-control"}))

    class Meta:
        model = get_user_model()
        fields = ['emp_id','first_name', 'last_name', 'email', 'username','password', 'address','gender']

class AdminForm(forms.ModelForm):
    first_name = forms.CharField(label="First Name", max_length=50, widget=forms.TextInput(attrs={"class": "form-control"}))
    last_name = forms.CharField(label="Last Name", max_length=50, widget=forms.TextInput(attrs={"class": "form-control"}))
    email = forms.EmailField(label="Email", max_length=50, widget=forms.EmailInput(attrs={"class": "form-control"}))
    username = forms.CharField(label="Username", max_length=50, widget=forms.TextInput(attrs={"class": "form-control"}))
    password = forms.CharField(label="Password", max_length=50, widget=forms.PasswordInput(attrs={"class": "form-control"}))
    class Meta:

        model = Admin
        fields=['first_name','last_name','email','username','password']

        
class DateForm(UserCreationForm,forms.Form):
    date = forms.DateField(label='Enter date')


class LoginForm(forms.Form):
    username = forms.CharField(label="Username", max_length=50, widget=forms.TextInput(attrs={"class": "form-control"}))
    password = forms.CharField(label="Password", max_length=50, widget=forms.PasswordInput(attrs={"class": "form-control"}))

    
    
