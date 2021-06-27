import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
my_video = cv2.VideoCapture(0)
while True:
    validation ,frame=my_video.read()
    if validation is not True:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, 1.3)
    for i,face in enumerate(faces):
        x, y, w, h = face
        image_face=frame[y:y+h,x:x+w]
        res_face=cv2.resize(image_face,(10,10))
        res_face = cv2.resize(res_face, (w,h))
        frame[y:y + h, x:x + w]=res_face

    cv2.imshow('img', frame)
    cv2.waitKey(10)