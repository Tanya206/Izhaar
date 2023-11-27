from django.shortcuts import render,redirect,HttpResponse
from .models import history
from django.contrib.auth.decorators import login_required

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
# Create your views here.

import tensorflow as tf
from keras.models import load_model
import pandas as pd
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
from cvzone.ClassificationModule import Classifier
import math
import os

# for real time translation by AJAX
from django.http import JsonResponse

# opencv to website
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
# space
from django.views.decorators.csrf import csrf_exempt
text=[]


def predict():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    folder = "Model"

    # Construct the full paths for the model and labels files
    model_path = os.path.join(folder, "keras_model.h5")
    labels_path = os.path.join(folder, "labels.txt")

    classifier = Classifier(model_path, labels_path)

    offset = 20
    imgSize = 300

    labels = ["A", "B", "C"]

    

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape 

            # Check if imgCrop is not empty and has valid dimensions before resizing
            if imgCrop.size != 0 and imgCropShape[0] > 0 and imgCropShape[1] > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
                text.append(labels[index])
                time.sleep(2.5)
                

            #cv2.imshow("ImageCrop", imgCrop)
            #cv2.imshow("ImageWhite", imgWhite)

        #cv2.imshow("Image", imgOutput)
        _, jpeg_image = cv2.imencode('.jpg', imgOutput)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg_image.tobytes() + b'\r\n\r\n')
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break
        # elif key== ord("w"):
        #     text.append(" ")
        #     break
        # elif key== ord("s"):
        #     text.append(".")
        #     break
    cap.release()
    cv2.destroyAllWindows()
        
    
    
    
        

# Release the camera and close all OpenCV windows
@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(predict(),content_type='multipart/x-mixed-replace;boundary=frame')
    except:
        pass
   # return render(request,'home.html',{})
@login_required
@csrf_exempt
def predictedtext(request):
    if request.method=='POST':
        save_history=request.POST.get('save_history')
        if save_history:
            user=request.user
            saved_text="".join(text)

            history.objects.create(user=user, translation_text=saved_text)
            text.clear()
            return redirect('show_history')

    s = "".join(text)
    context={'text': s}
    return JsonResponse(context)

def signToText(request):
    return render(request,'home.html')

@csrf_exempt  # This decorator is used to exempt CSRF protection for this view
def append_period(request):
    if request.method == 'POST':
        text.append(" ")  # Append space to the text variable
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})

@csrf_exempt
def append_space(request):
    if request.method == 'POST':
        text.append(". ")  # Append "." to the text variable
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})


def home1(request):
    context={}
    
    return render(request,'home1.html',context)

@login_required
def show_history(request):
    user=request.user
    history_enteries=history.objects.filter(user=user).order_by('-translation_date')
    return render (request, 'history.html',{'history_enteries':history_enteries})

# @login_required
# def save_history(request):
#     user=request.user
#     s="".join(text)

#     history.objects.create(user=user, translation_text=s)
#     text=[]
#     return
    # if request.method=='POST':
    #     translation_text=request.POST.get('translation_text','')
    #     user=request.user

    #     history.objects.create(user=user,translation_text=translation_text)


def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        fname=request.POST.get('fname')
        lname=request.POST.get('lname')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.first_name = fname
            my_user.last_name = lname
            my_user.save()
            messages.success(request, "Your account has been successfully created.")
            return redirect('login')
    return render (request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass1')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            fname = user.first_name
            return render(request, 'home.html',{'fname': fname})
        else:
            messages.error("Username or Password is incorrect!")

    return render (request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home1')