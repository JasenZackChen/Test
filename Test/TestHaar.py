import cv2
# 待检测的图片路径
import os
import numpy as np
imagepath = r'./1.jpg'
#建立相机
cap=cv2.VideoCapture(0)
# 获取训练好的参数数据，这里直接从GitHub上使用默认值
face_cascade = cv2.CascadeClassifier(r'../opencv2find/lbpcascades/lbpcascade_frontalface_improved.xml')
eyeglasses_cascade = cv2.CascadeClassifier(r'../opencv2find/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# 读取图片

#image = cv2.imread(imagepath)
while(1):
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 探测图片中的人脸
    eyeglasses=eyeglasses_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (5 ,5),
        flags = cv2.CASCADE_SCALE_IMAGE)

    print("发现{0}个人脸!".format(len(faces)))

    for(x ,y ,w ,h) in faces:
        # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
        cv2.circle(frame,(int(( x + x +w ) /2) ,int(( y + y +h ) /2)) ,int(w /2) ,(0,255,0) ,2)
    for (x, y, w, h) in eyeglasses:
        # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
        cv2.circle(frame, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 0, 255), 2)

    cv2.imshow("captrue",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

