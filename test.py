# import cv2
# import numpy as np

# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('pic.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Detect horizontal lines
# # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
# # horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# # Detect vertical lines
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
# vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# # plot vetical line into image



# # Combine masks and remove lines
# # table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
# # image[np.where(table_mask==255)] = [255,255,255]

# # cv2.imshow('thresh', thresh)
# # cv2.imshow('horizontal_mask', horizontal_mask)
# # cv2.imshow('vertical_mask', vertical_mask)
# # cv2.imshow('table_mask', table_mask)
# # cv2.imshow('image', image)
# cv2.waitKey()

# import cv2
# import numpy as np

# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('pic.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Detect vertical lines
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
# vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

# # Detect lines using HoughLinesP
# lines = cv2.HoughLinesP(vertical_mask, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=100)

# # Sort lines by their starting x-coordinate
# lines = sorted(lines, key=lambda line: line[0][0])
# print(len(lines))
# # Draw lines sequentially
# w=0
# h=0
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    

#     cv2.imshow("Image with Vertical Lines", image)
#     cv2.waitKey(0)  # Delay for visualization (adjust as needed)

# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('pic.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Detect vertical lines
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
# vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

# # Detect lines using HoughLinesP
# lines = cv2.HoughLinesP(vertical_mask, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=100)

# # Sort lines by their starting x-coordinate
# lines = sorted(lines, key=lambda line: line[0][0])

# # Initialize a variable to keep track of line index
# line_index = 0

# # Loop through the lines and crop between each pair of consecutive lines
# while line_index < len(lines) - 1:
#     line1 = lines[line_index]
#     line2 = lines[line_index + 1]

#     x1_1, y1_1, x2_1, y2_1 = line1[0]
#     x1_2, y1_2, x2_2, y2_2 = line2[0]

#     # Crop the region between the two lines
#     cropped_region = image[y2_1:y1_2, x1_1:x2_1]

#     # Check if the cropped region is not empty before saving
#     if not cropped_region.any():
#         print(f"Empty region between lines {line_index} and {line_index + 1}")
#     else:
#         # Save the cropped region to a file
#         cv2.imwrite(f"cropped_region_{line_index}.jpg", cropped_region)

#     # Increment the line index to process the next pair of lines
#     line_index +=1

# cv2.destroyAllWindows()




# import pytesseract
# from PIL import Image
# t=pytesseract.image_to_string(Image.open('shapes.png'),lang='eng', config='--psm 10')
# print(t)

# import keras_ocr
# import cv2
# img=cv2.imread("test.png")
# ocr_keras = keras_ocr.pipeline.Pipeline()
# t=ocr_keras.recognize([img])[0][0][0]
# print(t)
# import easyocr
# reader = easyocr.Reader(['ur'])
# text = reader.readtext('urdu.png')

import easyocr
import re

reader = easyocr.Reader(['ur'])
text_boxes = reader.readtext('urdu.png')
print(text_boxes)
for box in text_boxes:
    text = box[1]  # Extract the text from the box
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Urdu characters except spaces
    print(text)
# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Path to the Tesseract executable
# # pytesseract.pytesseract.TESSDATA_PREFIX = '/usr/share/tesseract-ocr/4.00/'  # Path to the tessdata directory
# # Load the image
# img = cv2.imread("shapes.png")

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Perform thresholding to improve the OCR accuracy
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Extract the text from the image
# text = pytesseract.image_to_string(thresh, lang="eng")

# # Print the extracted text
# print(text)
