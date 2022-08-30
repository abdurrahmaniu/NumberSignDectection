#improt all the library
from keras.models import load_model
import numpy as np
import cv2

model = load_model('model.h5')



cap = cv2.VideoCapture(0) #for video capture

while True:
    _,frame = cap.read() #frame variable er vitore data read hocce 
    cv2.rectangle(frame,(50,50),(250,300),(255,0,0),4)
    
    hand = frame[50:300,50:250]
    gray = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,135,255,cv2.THRESH_BINARY)
    #cv2.imshow('color',hand)
    cv2.imshow('Threshold',thresh)

    pred = thresh.reshape([1,250,200,1])/255
    cv2.putText(frame,str(np.argmax(model.predict(pred))),fontScale=2,org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=2,color=(0,255,0))
    cv2.imshow('Video',frame)
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break

    