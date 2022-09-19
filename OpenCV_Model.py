import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# IMPORTING TRAINED MODEL
model = keras.models.load_model(r'C:\OpenCV\Spot_Quality_Detection\Fruit_Quality_Recognition\Keras_Model\Fruit_Recognition_Model.h5')


#CAMERA ESTABLISHMENT

vid = cv2.VideoCapture(0)
print("Camera connection successfully established")

i = 0  # INITILIZING THE I VALUE FOR STARTING
while(True):  # CREATING WHILE LOOP FOR INFINITY LOOP 
    r, frame = vid.read()  # IT GIVES IMAGE OR FRAME FROM CAMERA
    cv2.imshow('frame', frame)   # IT IS SHOWING LIVE CAMERA FRAME 
    cv2.imwrite(r'C:\OpenCV\Spot_Quality_Detection\Fruit_Quality_Recognition\Cam_Images'+str(i)+".jpg", frame)  # IT IS STORING CAPTURED IMAGE IN CAME_IMAGE FOLDER
    test_image = tf.keras.preprocessing.image.load_img(r'C:\OpenCV\Spot_Quality_Detection\Fruit_Quality_Recognition\Cam_Images'+str(i)+".jpg", target_size = (224, 224)) # IT IS LOADING STORED IMAGE FROM CAM_IMAGES FOLDER
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)  # IT IS CONVERTING IMAGE TO ARRAY FROMATE
    test_image = np.expand_dims(test_image, axis = 0)  # IT IS EXPANDING DIMENTIONS OF IMAGE LIKE (1,224,224,3)

    Predictions=model.predict(test_image) # IT IS PREDICTING THE GIVE IMAGE
    Max_val=np.argmax(Predictions) # IT GIVES THE MAX_VALUE FROM PREDICTIONS
    classes={0: 'fresh_apples', 1: 'fresh_banana', 2: 'fresh_oranges', 3: 'rotten_apples', 4: 'rotten_banana', 5: 'rotten_oranges'}  # IT GIVES CLASS NAMES 
    print('Classes_Names :',classes) # IT IS SHOWING GIVES CLASS NAMES
    print('Max_Prob_val: ',Max_val) # IT IS SHOWING MAXIMUM PROBOBILITY VALUE FROM PREDICTIONS
    print('Predicted Fruit :',classes[Max_val]) # IT SHOWS PREDICTED FRUIT NAME

    cv2.putText(frame, classes[Max_val], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)  # IT IS MAKING TEXT ON IMAGE , MEANS PREDICTED FRUIT NAME SHOWS ON FRAME OR IMAGE 
    cv2.imshow('Predicted Image',frame)  # IT SHOWS THE PREDICTED IMAGE
    os.remove(r'C:\OpenCV\Spot_Quality_Detection\Fruit_Quality_Recognition\Cam_Images'+str(i)+".jpg") #  IT IS REMOVING IMAGE WHICH WAS SOTERD IN CAM IMAGE FOLDER
    i = i + 1 # IT IS FOR INCREMENTATION
    if cv2.waitKey(1) & 0xFF == ord('q'):  # IT IS FOR WHEN YOU PRESS 'q' KEY BUTTEN ON KEY BOARD THE WHILE LOOP WILL GET STOP (BREAK)
        break
vid.release() # IT RELEASE THE CAMERA
cv2.destroyAllWindows() # IT DISTROY ALL WINDOWS