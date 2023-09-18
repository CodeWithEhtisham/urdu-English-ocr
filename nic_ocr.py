import numpy as np
import cv2

img = cv2.imread('pic.jpeg')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

i=0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    # cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    # if len(approx) == 3:
    #     cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
    if len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        # w,h,x,y=71 ,66 ,1127 ,12
        if (w and h)>10 and (w and h)<100:
            print(w, h,i)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
            cv2.imwrite("shapes_{0}.png".format(i), img)
            i+=1
        if i == 7:
            break
        
            # break
        # if aspectRatio >= 0.95 and aspectRatio < 1.05:
        #     cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # else:
        #     cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    # elif len(approx) == 5 :
    #     cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # elif len(approx) == 10 :
    #     cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # else:
    #     cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import numpy as np
# import cv2

# img = cv2.imread('pic.jpeg')
# imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thrash = cv2.threshold(imgGry, 240, 255, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# i = 0
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1] - 5
#     if len(approx) == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w) / h
#         if (w and h) > 10 and (w and h) < 100:
#             print(w, h, i)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
#             cv2.imwrite("shapes_{0}.png".format(i), img)
#             i += 1
#     if i == 7:
#         break

# # Display the processed image with rectangles
# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
