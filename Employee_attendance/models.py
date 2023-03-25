from django.db import models
from django.contrib.auth.models import AbstractUser, User


class User(AbstractUser):
    user_type_data=((1,"Admin"),(2,"Employee"))
    user_type=models.CharField(default=1,choices=user_type_data,max_length=10)


class Employee(models.Model):
    user=models.OneToOneField('User',on_delete=models.CASCADE,primary_key=True)
    emp_id=models.IntegerField()
    gender=models.CharField(max_length=255,default=None)
    address=models.TextField()

class Attendance(models.Model):
    emp_id=models.ForeignKey(Employee,on_delete=models.DO_NOTHING)
    date = models.DateField(auto_now_add=True)
    status = models.BooleanField(default=False)

class Admin(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
