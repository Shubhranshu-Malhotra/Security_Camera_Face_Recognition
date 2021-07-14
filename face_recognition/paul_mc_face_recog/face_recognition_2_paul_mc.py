import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
import time
import os
import random
print("OpenCV version: ", cv2.__version__)
print("Imports Successful!!")

train_image_dir = r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known\\"
test_image_dir = r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\unknown\\"

train_images_fnames = [img for img in os.listdir(train_image_dir)]
test_images_fnames = [img for img in os.listdir(test_image_dir)]


# Load images and encode them

time_1 = time.time()
all_train_encodings = []
all_names = []

for fname in train_images_fnames:
    train_img = fr.load_image_file(train_image_dir + fname)
    train_face_position = fr.face_locations(train_img)
    img_encoding = fr.face_encodings(train_img, train_face_position)[0]
    all_train_encodings.append(img_encoding)
    all_names.append(fname.split(sep = '.')[0])

time_2 = time.time()
print("Time to encode = ", time_2 - time_1)


# Testing the program

def get_random_prediction(test_image_dir, test_images_fnames, train_labels, train_encodings):
    """
    Takes in test image directory, test image fnames, training labels and training encodings
    Returns a prediction on a random image from the given folder with recognized face.
    """
    random_fname = random.choice(test_images_fnames)
    time_3 = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    test_image = fr.load_image_file(test_image_dir + random_fname)
    test_face_positions = fr.face_locations(test_image)
    test_encodings =  fr.face_encodings(test_image, test_face_positions)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left), face_encoding in zip(test_face_positions, test_encodings):
        name = 'Unknown'
        matches = fr.compare_faces(train_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = train_labels[first_match_index]
        cv2.rectangle(test_image, (left, top), (right,bottom), (0,255,0), thickness = 2)
        cv2.putText(test_image, name, (left,top-6), font, fontScale = 0.75, color = (190,120,), thickness = 2)
    time_4 = time.time()

    print("Time to predict on test image: ", time_4 - time_3)
    # Show the image
    cv2.imshow(f"{random_fname}", test_image)
    cv2.moveWindow(f"{random_fname}", 0, 0)
    if cv2.waitKey(0) == ord('q'):    
        cv2.destroyAllWindows()

get_random_prediction(test_image_dir, test_images_fnames, all_names, all_train_encodings )

