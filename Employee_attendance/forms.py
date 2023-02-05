from django import forms
from .models import *

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            'emp_id',
            'name',
            'address',
            'phone',
            'email',
            'username',
            'password',
            # 'image'
            ]
    
class AttendanceForm(forms.ModelForm):
    class Meta:
        model=Attendance
        fields=[
            'emp_id',
            'status'
        ]

class AdminForm(forms.ModelForm):
    class Meta:
        model=Admin
        fields=[
            'username',
            'password'
        ]