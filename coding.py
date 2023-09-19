import numpy as np
import cv2
import easyocr
reader = easyocr.Reader(['ch_sim','en']) 
i=0

def english_ocr(img):
    try:
        text = reader.readtext(img)
        return text
    except Exception as e:
        return None
img = cv2.imread('pic.jpeg')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

sorted_contours =sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

for contour in sorted_contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    # cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        if (w and h)>10 and (w and h)<100:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
            # cv2.imwrite("shapes_{0}.png".format(i), img)
            # crop region of img using bounding box
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite("shapes.png", crop_img)
            # call english ocr function
            text = english_ocr('shapes.png')
            print(text)
            i+=1
        # if i==13:
        #     break
        # if i%7 == 0:
        #     i=0


cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()