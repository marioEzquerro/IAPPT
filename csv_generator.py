from tkinter import image_names
from matplotlib.pyplot import figure
import numpy as np
import os
import cv2
import mediapipe as mp
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

PATH = ['.\\data\\train\\paper', '.\\data\\train\\scissors', '.\\data\\train\\rock']
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Añadir todas las imagenes a la lista
imagenes = []
for folder in PATH:
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if ".DS_Store" not in full_path:
            imagenes.append(full_path)
 


# Creamos un objeto del tipo ImageDataGenerator para transformar imagenes
datagen = ImageDataGenerator(
    rotation_range=180,
    horizontal_flip=True,
    fill_mode='nearest')

# -- AUGMENTACION -----------------
# test_img = cv2.imread(IMAGE_FILES[0]) # Elegir imagen para transformar
# img = img_to_array(test_img)  # convert image to numpy arry
# img = img.reshape((1,) + img.shape)  # reshape image

# i = 0
# # bach es cada una de las imagenes augmentadas
# for batch in datagen.flow(img, save_to_dir='aug', save_prefix='test_', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
#     i += 1
#     if i > 1:  # Aplicar n veces por imagen
#         break

# ---------------------------------

csv = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(imagenes):
        # Read an image, flip it around y-axis for correct handedness output (see above)
        img = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if 'rock' in file:
            figura  = 1
        elif 'paper' in file:
            figura  = 2
        elif 'scissors' in file:
            figura  = 3
        else: 
            figura = 0

        # Print handedness and draw hand landmarks on the image
        if results.multi_hand_landmarks:
            mano = results.multi_hand_landmarks[0]
            
            muñecaY = mano.landmark[0].y
            muñecaX = mano.landmark[0].x
            indxpY = mano.landmark[6].y  
            indxpX = mano.landmark[6].x
            indxtY = mano.landmark[8].y  
            indxtX = mano.landmark[8].x
            corzpY = mano.landmark[10].y  
            corzpX = mano.landmark[10].x
            corztY = mano.landmark[12].y  
            corztX = mano.landmark[12].x
            anlpY = mano.landmark[14].y  
            anlpX = mano.landmark[14].x
            anltY = mano.landmark[16].y  
            anltX = mano.landmark[16].x
            meñpY = mano.landmark[18].y  
            meñpX = mano.landmark[18].x
            meñtY = mano.landmark[20].y  
            meñtX = mano.landmark[20].x
            
            csv.append([figura,muñecaX, muñecaY,indxpY,indxpX,indxtY,indxtX,corzpY,corzpX,corztY,corztX,anlpY,anlpX,anltY,anltX,meñpY,meñpX,meñtY,meñtX])
            
# Guardar todos los datos en csv
np.savetxt("GFG.csv", csv, delimiter =", ", fmt ='% s')