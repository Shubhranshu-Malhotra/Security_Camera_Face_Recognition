import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
import time
import os
import random
import pickle
print("OpenCV version: ", cv2.__version__)
print("Imports Successful!!")

train_image_dir = r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known"

train_images_fnames = [img for img in os.listdir(train_image_dir)]

# Load images and encode them

time_1 = time.time()
Encodings = []
Names = []

for fname in train_images_fnames:
    train_img = fr.load_image_file(os.path.join(train_image_dir, fname))
    train_face_position = fr.face_locations(train_img)
    img_encoding = fr.face_encodings(train_img, train_face_position)[0]
    Encodings.append(img_encoding)
    Names.append(fname.split(sep = '.')[0])
    print("Encoded: ", os.path.join(train_image_dir, fname))

time_2 = time.time()
print("Time to encode = ", time_2 - time_1)

with open('train.pkl','wb') as f:
    pickle.dump(Names, f)
    pickle.dump(Encodings, f)

