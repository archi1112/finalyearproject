from django.db import models
from django.contrib.auth.models import AbstractUser, User
from django.db.models.signals import post_save
# from django import forms
from django.dispatch import receiver

class User(AbstractUser):
    user_type_data=((1,"Admin"),(2,"Employee"))
    user_type=models.CharField(default=1,choices=user_type_data,max_length=10)


class Employee(models.Model):
    user=models.OneToOneField('User',on_delete=models.CASCADE,primary_key=True)
    emp_id=models.IntegerField()
    gender=models.CharField(max_length=255,default=None)
    address=models.TextField()
    # city=models.TextField()
    # phoneRegex = RegexValidator(regex = r"^\+?1?\d{8,15}$")
    # phone = models.CharField(validators = [phoneRegex], max_length = 16, unique = True)


class Attendance(models.Model):
    emp_id=models.ForeignKey(Employee,on_delete=models.DO_NOTHING)
    date = models.DateField(auto_now_add=True)
    status = models.BooleanField(default=False)


class Admin(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
