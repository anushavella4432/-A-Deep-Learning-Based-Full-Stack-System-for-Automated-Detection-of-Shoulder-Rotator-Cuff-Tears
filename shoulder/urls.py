"""shoulder URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
"""
URL configuration for radiotherapy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from admins import views as a
from users import views as v
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',a.index,name='index'),
    path('about/',a.about, name='about'),
    path('adminlogin/',a.adminlogin, name='adminlogin'),
    path('contact/',a.contact, name='contact'),
    path('feature/',a.feature, name='feature'),
    path('service/',a.service, name='service'),
    path('register/',v.register, name='register'),
    path('userlogin/',v.userlogin, name='userlogin'),
    path('dashboard/',a.dashboard, name='dashboard'),
    path('appointment/',a.appointment,name='appointment'),
    path('upload/',a.upload,name='upload'),
    
    path('resnet/',a.resnet,name='resnet'),
    path('vgg16/',a.vgg16,name='vgg16'),


   path('udashboard/',v.udashboard, name='udashboard'),
   path('prediction/',v.prediction, name='prediction'),
   path('profile/',v.profile, name='profile'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


