from django import forms
from .models import *

class ResgistrationForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'face_id',
            'name',
            'address',
            'phone',
            'email',
            'username',
            'password',
            # 'image'
            ]
    
class Attendance(forms.ModelForm):
    class Meta:
        model=attend
        fields=[
            'face_id',
        ]

class Admin(forms.ModelForm):
    class Meta:
        model=Admin
        fields=[
            'username',
            'password'
        ]