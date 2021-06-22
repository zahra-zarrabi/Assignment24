import random
import cv2

face_detector=cv2.CascadeClassifier('faceshape.xml')
image=cv2.imread('hamegi.jpg')
gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cords = face_detector.detectMultiScale(gray_image,1.3)

faces=[]
for cord in cords:
    x,y,w,h=cord
    image_face = image[y:y + h, x:x + w]
    faces.append(image_face)


for cord in cords:
    x, y, w, h = cord
    sticker = random.choice(faces)
    re_sticker = cv2.resize(sticker, (w, h))
    image[y:y + h, x:x + w]=re_sticker

cv2.imshow('out.jpg',image)
cv2.waitKey()