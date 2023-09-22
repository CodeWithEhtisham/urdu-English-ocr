import cv2
import numpy as np
import easyocr
import re
reader = easyocr.Reader(['ch_sim','en']) 
def image_to_text(img):
        # get roi of each line
    try:
        text = reader.readtext(img)[0][1]
        # cv2.imshow('shapes', img)
        # cv2.waitKey(0)
        return text
    except Exception as e:
        return None

def preprocess_column(img):
    ls=[]
    imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
    contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    i=0
    # sort coutours with respect to sequence and no skip
    sorted_contours =sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])
    for contour in sorted_contours:
        approx = cv2.approxPolyDP(contour, 0.02* cv2.arcLength(contour, True), True)
        # cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        # if len(approx) == 3:
        #     cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
        # print(f"i value is {i}", len(approx))
        if len(approx) == 4 :
            x, y , w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            # w,h,x,y=71 ,66 ,1127 ,12
            # print(w, h,i)
            if (w and h)>6:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
                # cv2.imwrite("shapes_{0}.png".format(i), img)
                roi = img[y:y+h, x:x+w]
                text = image_to_text(roi)
                ls.append(text)
    # print(ls)
    # pattern = re.compile(r'^\d+$|^\w+-\w+-\w+$')
    pattern=re.compile(r'\d')

    # Extract values matching the pattern and remove None values
    # print(ls)
    values = [x for x in ls if x is not None and pattern.match(x)]

    return values

# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('pic.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

# Detect lines using HoughLinesP
lines = cv2.HoughLinesP(vertical_mask, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=100)

# Sort lines by their starting x-coordinate
lines = sorted(lines, key=lambda line: line[0][0])

# Initialize a list to store the cropped images
cropped_images = []

# Get the width of the original image
image_width = image.shape[1]

# Iterate through the sorted lines
for idx, line in enumerate(lines):
    x1, _, x2, _ = line[0]

    # Check if the cropped part has a valid width
    if x1 > 0 and x1 < image_width:
        # Crop the image to the left of the current vertical line
        if idx == 0:
            cropped_part = image[:, :x1]
        else:
            prev_x2, _, _, _ = lines[idx - 1][0]
            cropped_part = image[:, prev_x2-10:x1+10]

        # Append the cropped part to the list
        cropped_images.append(cropped_part)

# Check if the part to the right of the last vertical line has a valid width
if lines:
    x1, _, _, _ = lines[-1][0]
    if x1 > 0 and x1 < image_width:
        cropped_images.append(image[:, x1:])
vertical_image_list=[]
# Display and save the cropped images
for idx, cropped_part in enumerate(cropped_images):
    if cropped_part.shape[1] > 30:  # Check if width is greater than zero
        # cv2.imshow(f"Cropped Part {idx}", cropped_part)
        # cv2.imwrite(f"cropped_part_{idx + 1}.jpg", cropped_part)
        # print(f"Cropped Part {idx}")
        vertical_image_list.append(cropped_part)
# print(len(vertical_image_list))
dc={}
for idx, i in enumerate(vertical_image_list):
    ls=preprocess_column(i)
    
    if len(ls)<10:
        continue
    # print(len(ls))
    # print(ls)
    dc[idx]=ls

max_len=[]
for i in dc.values():
    max_len.append(len(i))

max_len=max(max_len)
for i in dc.values():
    while len(i)<max_len:
        i.append(" ")

import pandas as pd
df=pd.DataFrame.from_dict(dc)
df.to_csv("output.csv")
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('s_no.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Detect horizontal lines
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
# horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# # Detect lines using HoughLinesP for horizontal lines
# horizontal_lines = cv2.HoughLinesP(horizontal_mask, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=100)

# # Check if horizontal lines were detected
# if horizontal_lines is not None:
#     # Initialize a copy of the image for drawing lines
#     image_with_lines = image.copy()

#     # Sort horizontal lines by their starting y-coordinate
#     horizontal_lines = sorted(horizontal_lines, key=lambda line: line[0][1])

#     # Initialize an index to keep track of the current line
#     current_line_idx = 0

#     while current_line_idx < len(horizontal_lines):
#         line = horizontal_lines[current_line_idx]
#         _, y1, _, y2 = line[0]

#         # Draw the current line on the image
#         cv2.line(image_with_lines, (0, y1), (image.shape[1], y2), (0, 0, 255), 2)

#         # Display the image with the current line
#         cv2.imshow("Image with Horizontal Line", image_with_lines)

#         # Wait for a key press (q to move to the next line, any other key to continue drawing)
#         key = cv2.waitKey(0)

#         if key == ord('q'):
#             current_line_idx += 1

#     cv2.destroyAllWindows()

# else:
#     print("No horizontal lines detected.")

