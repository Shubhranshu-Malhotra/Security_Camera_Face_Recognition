import cv2
import wget
import time
print("imports successful!!")

# download the cascade
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# wget.download(url)

# Load cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from security cam. 
ip = "***.***.***.***"
username = "***"
password = "******"
channel = 1
subtype = 0
cam = cv2.VideoCapture(f'rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}')
print('Is the IP camera turned on: {}'.format(cam.isOpened()))

while True:
    
    # Time to process one frame

    start = time.time()
    # Read the frame
    _, frame = cam.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # End time
    end = time.time()
    time_taken = end - start
    fps = 1/time_taken
    cv2.rectangle(frame, (0, 0), (80, 50), (0, 0, 255), 3)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{fps:.1f}", (20,20),fontFace = font, fontScale = 0.5, color = (0,0,255))
    # Display
    cv2.imshow('Face Detection', frame)
    cv2.moveWindow('Face Detection',0,0)

    # Stop if escape key is pressed
    if cv2.waitKey(1)==ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()