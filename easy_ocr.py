import cv2
import numpy as np

#Read input image
img = cv2.imread('pic.jpeg')

#convert from BGR to HSV color space
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#apply threshold
thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

# find contours and get one with area about 180*35
# draw all contours in green and accepted ones in red
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
#area_thresh = 0
min_area = 0.95*180*44
max_area = 1.05*180*44
print('mix',min_area)
print('max',max_area)
result = img.copy()
i = 1
for c in contours:
#     print(c)
    area = cv2.contourArea(c)
    cv2.drawContours(result, [c], -1, (0, 255, 0), 1)
    
    x,y,w,h = cv2.boundingRect(c)
    # crop region of img using bounding box
    region = result[y:y+h, x:x+w]
    # save region to new image
    print(region.shape,' i ',i)
#     cv2.imwrite("black_region_{0}.png".format(i), region)
    i = i + 1

    # if region.shape[0]>150 and region.shape[1]<50:
    #     cv2.imwrite("black_region_{0}.png".format(i), region)
    #     break
#     if area > min_area and area < max_area:
#             cv2.drawContours(result, [c], -1, (0, 0, 255), 1)
#             break

# save result
# cv2.imwrite("box_found.png", result)

# show images
# cv2.imshow("GRAY", gray)
# cv2.imshow("THRESH", thresh)
cv2.imshow("RESULT", result)
cv2.waitKey(0)