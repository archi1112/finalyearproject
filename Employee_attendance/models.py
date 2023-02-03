from django.db import models
from django.db.models.signals import post_save

class UserProfile(models.Model):
    face_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)
    address = models.CharField(max_length = 100)
    phone = models.CharField(max_length =  10)
    email = models.EmailField(max_length = 20)
    username=models.CharField(max_length=21)
    password=models.CharField(max_length=20)
    # image = models.ImageField(upload_to='profile_image', blank=True)


class attend(models.Model):
    face_id=models.ForeignKey(UserProfile,on_delete=models.CASCADE,)
    present=models.BooleanField()
