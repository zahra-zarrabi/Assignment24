import random
import cv2

mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

my_video = cv2.VideoCapture(0)
img_red=cv2.imread('image/red_lipstick.png',cv2.IMREAD_UNCHANGED)
img_pink=cv2.imread('image/pink_lipstick.png',cv2.IMREAD_UNCHANGED)
img_green=cv2.imread('image/green_eye.png',cv2.IMREAD_UNCHANGED)

while True:
    validation ,frame=my_video.read()
    if validation is not True:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(frame_gray, 1.3)
    for eye in eyes:
        x, y, w, h = eye

        sticker=img_green
        res_sticker=cv2.resize(sticker,(w,h))
        for m in range(h):
            for n in range(w):
                if res_sticker[m, n, 3] != 0:
                    frame[y + m, x + n] = res_sticker[m, n, 0:3]


    mouths = mouth_cascade.detectMultiScale(frame_gray,1.3,10)
    for mouth in mouths:
        ex, ey, ew, eh=mouth

        sticker = random.choice([img_pink,img_red])
        res_sticker = cv2.resize(sticker, (ew, eh))
        for m in range(eh):
            for n in range(ew):
                if res_sticker[m, n, 3] != 0:
                    frame[ey + m, ex + n] = res_sticker[m, n, 0:3]



    cv2.imshow('img',frame)
    cv2.waitKey(10)
