import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
import time
print("OpenCV version: ", cv2.__version__)
print("Imports Successful!!")

# Load and encode image
time_1 = time.time()
donald_img = fr.load_image_file("D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known\Donald Trump.jpg")
donald_encode = fr.face_encodings(donald_img)[0]
chase_img = fr.load_image_file("D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known\Chase.jpg")
chase_encode = fr.face_encodings(chase_img)[0]
nancy_img = fr.load_image_file("D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known\\Nancy Pelosi.jpg")
nancy_encode = fr.face_encodings(nancy_img)[0]
pence_img = fr.load_image_file("D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known\Mike Pence.jpg")
pence_encode = fr.face_encodings(pence_img)[0]

time_2 = time.time()

print("Time to encode = ", time_2 - time_1)
Encodings = [donald_encode, nancy_encode, chase_encode, pence_encode]
Names = ['Donald Trump', 'Nancy Pelosi', 'Chase', 'Mike Pence']

# Testing the program
time_3 = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
test_image = fr.load_image_file(r'D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\unknown\u11.jpg')
test_face_positions = fr.face_locations(test_image)
all_test_encodings =  fr.face_encodings(test_image, test_face_positions)

test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(test_face_positions, all_test_encodings):
    name = 'Unknown'
    print(len(face_encoding), len(Encodings[0]), len(Encodings))
    matches = fr.compare_faces(Encodings, face_encoding)
    print(matches, len(matches))
    if True in matches:
        first_match_index = matches.index(True)
        name = Names[first_match_index]
    cv2.rectangle(test_image, (left, top), (right,bottom), (0,255,0), thickness = 2)
    cv2.putText(test_image, name, (left,top-6), font, fontScale = 0.75, color = (190,120,), thickness = 2)
time_4 = time.time()

print("Time to predict on test image: ", time_4 - time_3)
print("Total Time: ", time_4 - time_1)

# Show the image
cv2.imshow("window", test_image)
cv2.moveWindow("window", 0, 0)
if cv2.waitKey(0) == ord('q'):    
    cv2.destroyAllWindows()


