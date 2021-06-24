import random
import cv2

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
my_video=cv2.VideoCapture(0)

while True:
    validation ,frame=my_video.read()
    if validation is not True:
        break
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cords = face_detector.detectMultiScale(frame_gray,1.3)

    faces=[]
    for cord in cords:
        x,y,w,h=cord
        image_face = frame[y:y + h, x:x + w]
        faces.append(image_face)


    for cord in cords:
        x, y, w, h = cord
        sticker = random.choice(faces)
        re_sticker = cv2.resize(sticker, (w, h))
        frame[y:y + h, x:x + w]=re_sticker

    cv2.imshow('output.jpg',frame)
    cv2.waitKey(10)