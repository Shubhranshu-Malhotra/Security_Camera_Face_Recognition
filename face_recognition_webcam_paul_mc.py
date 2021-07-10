import face_recognition
import cv2
import os
import pickle
import time
print(cv2.__version__)
 
Encodings=[]
Names=[]
 
with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
print(len(Encodings))
print(Names, len(Names))
font=cv2.FONT_HERSHEY_SIMPLEX

# cam= cv2.VideoCapture(0)

ip = "***.***.***.***"
username = "***"
password = "******"
channel = 1
subtype = 0
cam = cv2.VideoCapture(f'rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}')
print('Is the IP camera turned on: {}'.format(cam.isOpened()))

while True:
 
    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
        top=top*4
        right=right*4
        bottom=bottom*4
        left=left*4
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()