import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
import time
import os
import random
print("OpenCV version: ", cv2.__version__)
print("Imports Successful!!")

train_image_dir = r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\known"
test_image_dir = r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\paul_mc_images\unknown"

print(os.path.join(train_image_dir, "paul"))