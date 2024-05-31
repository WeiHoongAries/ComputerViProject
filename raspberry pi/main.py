import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
image = cv2.imread('raspberry pi//pen15.jpg',cv2.IMREAD_GRAYSCALE)
blur=cv2.GaussianBlur(image,(5,5),0)
_, after_thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# dist = cv2.distanceTransform(after_thresh,cv2.DIST_L2,3)
# dist = cv2.normalize(dist,dist,0,1.0,cv2.NORM_MINMAX)
# dist = (dist*255).astype("uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
opening=cv2.morphologyEx(after_thresh,cv2.MORPH_OPEN,kernel)
text = pytesseract.image_to_string(opening, config='--psm 6')
print(text)
cv2.imshow("image",opening)
cv2.waitKey(0)
cv2.destroyAllWindows