from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('',views.home1,name="home1"),
    path('signup/',views.SignupPage,name='signup'),
    path('login/',views.LoginPage, name='login'),
    path('logout/',views.LogoutPage,name='logout'),
]