import face_recognition as fr
import cv2
print("Opencv version: ", cv2.__version__)
print("Imports Successful!!")

# Load image to find face in
temp_img = fr.load_image_file("D:\Projects\security_camera_face_recognition\paul_mcwhorter_images\known\Chase.jpg")

# Find the location of the faces using face_recognition
face_locs = fr.face_locations(temp_img)
print(face_locs)

# Convert from RGB to BGR to work using opencv
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

# Draw rectangles arount the found faces
for (row1, col1, row2, col2) in face_locs:
    temp_img = cv2.rectangle(temp_img, (col1, row1), (col2,row2), (0,255,0), thickness = 2)

# Show the image
cv2.imshow("temp_window", temp_img)
cv2.moveWindow("temp_window", 0, 0)
if cv2.waitKey(0) == ord('q'):    
    cv2.destroyAllWindows()
