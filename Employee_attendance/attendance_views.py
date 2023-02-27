from django.shortcuts import render, redirect
from .forms import *
from django.contrib import messages
from datetime import date
from django.contrib.auth.models import User
from .models import Attendance


class AttendanceManager:
    
    def attendance_by_date(request):
        if request.method == 'POST':
            form = DateForm(request.POST)
            if form.is_valid():
                date = form.cleaned_data['date']
                attendance = Attendance.objects.filter(date=date)
                context = {
                    'attendance': attendance,
                    'date': date,
                }
                return render(request, 'attendanceDetails.html', context)
        else:
            form = DateForm()

        context = {
            'form': form,
        }
        return render(request, 'admin_home.html', context)
    
    def markAttendance(self, emp_id):
        if emp_id:
            employee = Employee.objects.get(emp_id=emp_id)
            attendance, created = Attendance.objects.get_or_create(
                emp_id=employee, date=date.today())
            if not attendance.status:
                attendance.status = True
                attendance.save()

    def update_attendance(self):
        pass
