import cv2
import random

fac_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
my_video=cv2.VideoCapture('image/mm.mp4')

sti1=cv2.imread('image/sticker1.png',cv2.IMREAD_UNCHANGED)
sti2=cv2.imread('image/sticker2.png',cv2.IMREAD_UNCHANGED)
sti3=cv2.imread('image/sticker3.png',cv2.IMREAD_UNCHANGED)

while True:
    validation ,frame=my_video.read()
    if validation is not True:
        break
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces=fac_detector.detectMultiScale(frame_gray,1.5)
    for i ,face in enumerate(faces):
        x,y,w,h=face
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),8)
        sticker = random.choice([sti1, sti2, sti3])
        res_sticker = cv2.resize(sticker, (w, h))
        for m in range(h):
            for n in range(w):
                if res_sticker[m, n, 3] != 0:
                    frame[y + m, x + n] = res_sticker[m, n, 0:3]

    cv2.imshow('output.jpg',frame)
    cv2.waitKey(10)