from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django import forms


class Employee(models.Model):
    emp_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)
    address = models.CharField(max_length = 100)
    phone = models.CharField(max_length =  10)
    email = models.EmailField(max_length = 20)
    username=models.CharField(max_length=21)
    password=models.CharField(max_length=20)
    objects=models.Manager()
    # image = models.ImageField(upload_to='profile_image', blank=True)


class Attendance(models.Model):
    emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date=models.DateTimeField(auto_now_add=True)
    status=models.BooleanField(default=False)
    objects=models.Manager()

class Admin(models.Model):
    username=models.CharField( max_length=50)
    password=models.CharField(max_length=20)
    objects=models.Manager()




