from django.shortcuts import render,redirect,HttpResponse
from django.contrib.auth.forms import UserCreationForm

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
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
#from django.views.decorators.csrf import csrf_exempt
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

def predictedtext(request):
    context={'text': text}
    return JsonResponse(context)

def signToText(request):
    return render(request,'home.html')

def append_space(request):
    # Example: Append a space to the text variable on the server
    if request.method =='POST' and request.is_ajax():
        print("Hello")
        text.append(" ")
        return JsonResponse({"status": "success"})

def append_period(request):
    # Example: Append a period to the text variable on the server
    if request.method =='POST' and request.is_ajax():
        text.append(".")
        return JsonResponse({"status": "success"})



def home1(request):
    context={}
    
    return render(request,'home1.html',context)


def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')
    return render (request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home1')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home1')