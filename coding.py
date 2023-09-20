import numpy as np
import cv2
import easyocr
import re

reader = easyocr.Reader()
import keras_ocr
ocr_keras = keras_ocr.pipeline.Pipeline()
img = cv2.imread('pic.jpeg')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
i=0

def english_orcr(img):
    try:
        cv2.imshow('shapes', img)
        cv2.waitKey(0)
        t=ocr_keras.recognize([img])[0][0][0]
        print(t)
        return t
    except Exception as e:
        print(e)
        
ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
sorted_contours =sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])
s_no,h_no,name,f_name,cnic,age,adress=[],[],[],[],[],[],[]
for contour in sorted_contours:
    approx = cv2.approxPolyDP(contour, 0.04* cv2.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        # w,h,x,y=71 ,66 ,1127 ,12
        if (w and h)>25 and (w and h)<150:
            # print(w, h,i)
            # if i in [2,3,6]:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
            #     # cv2.imwrite("shapes_{0}.png".format(i), img)
            #     # print(i)
            #     roi=img[y:y+h,x:x+w]
            #     if i ==2
            # i+=1
            roi=img[y:y+h,x:x+w]
            if i==0:
                # text = english_orcr(roi)
                s_no.append(roi)
            elif i==1:
                # text = english_orcr(roi)
                h_no.append(roi)
            elif i==2:
                # text = english_orcr(roi)
                name.append(roi)
            elif i==3:
                # text = english_orcr(roi)
                f_name.append(roi)
            elif i==4:
                # text = english_orcr(roi)
                cnic.append(roi)
            elif i==5:
                # text = english_orcr(roi)
                age.append(roi)
            elif i==6:
                # text = english_orcr(roi)
                adress.append(roi)
            i+=1
            if i==7:
                i=0

        # if i == 13:
        #     break
import pandas as pd
for i in s_no:
    english_orcr(i)
# print(s_no)
# df = pd.DataFrame({'s_no':s_no,'h_no':h_no,'name':name,'f_name':f_name,'cnic':cnic,'age':age,'adress':adress})
        
# cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
