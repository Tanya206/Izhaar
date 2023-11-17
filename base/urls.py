from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('',views.home1,name="home1"),
    path('signup/',views.SignupPage,name='signup'),
    path('login/',views.LoginPage, name='login'),
    path('logout/',views.LogoutPage,name='logout'),
    path('predict/',views.signToText,name="predict"),
    path('video_feed/',views.video_feed,name="video_feed"),
    path('predictedText/',views.predictedtext,name="predicted_text"),
    path('append_period/',views.append_period,name="append_period"),
    path('append_space/', views.append_space,name="append_space"),

]