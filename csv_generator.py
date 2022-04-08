import numpy as np
import os
import cv2
import mediapipe as mp
from keras.preprocessing.image import ImageDataGenerator

mp_hands = mp.solutions.hands
TRAINING_FOLDERS = ['.\\data\\train\\paper', '.\\data\\train\\scissors', '.\\data\\train\\rock']
TESTIING_FOLDERS = ['.\\data\\test\\paper', '.\\data\\test\\scissors', '.\\data\\test\\rock']
images = []

# Creamos un objeto del tipo ImageDataGenerator para transformar imagenes
# datagen = ImageDataGenerator(
#     rotation_range=180,
#     horizontal_flip=True,
#     fill_mode='nearest')

def loadTrainingImages():
    # Añadir todas las imagenes a la lista
    for folder in TRAINING_FOLDERS:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if ".DS_Store" not in full_path:
                images.append(full_path)
    saveData('training_data.csv')
    
def loadTestingImages():
    # Añadir todas las imagenes a la lista
    for folder in TESTIING_FOLDERS:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if ".DS_Store" not in full_path:
                images.append(full_path)
    saveData('testing_data.csv')
    

def saveData(name):
    print('Generando csv...')
    csv = []
    with mp_hands.Hands(
            static_image_mode = True,
            model_complexity = 0,
            max_num_hands = 1,
            min_detection_confidence = 0.5) as hands:

        for idx, file in enumerate(images):
            print(f'Convertiendo: {file}')
            # Leer imagen y voltear horizontalmente 
            img = cv2.flip(cv2.imread(file), 1)
            # Convertir de BGR a RGB antes de procesar
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if 'rock' in file:
                figura = 1
            elif 'paper' in file:
                figura = 2
            elif 'scissors' in file:
                figura = 3
            else: 
                figura = 0

            # Print handedness and draw hand landmarks on the image
            if results.multi_hand_landmarks:
                # Usamos solo el primer elemento del array que contiene las manos encontradas
                mano = results.multi_hand_landmarks[0]
            
                csv.append([
                    figura, mano.landmark[0].y, mano.landmark[0].x, 
                    mano.landmark[6].y, mano.landmark[6].x,
                    mano.landmark[8].y, mano.landmark[8].x, 
                    mano.landmark[10].y, mano.landmark[10].x, 
                    mano.landmark[12].y, mano.landmark[12].x, 
                    mano.landmark[14].y, mano.landmark[14].x,
                    mano.landmark[16].y, mano.landmark[16].x,
                    mano.landmark[18].y, mano.landmark[18].x, 
                    mano.landmark[20].y, mano.landmark[20].x
                ])
    # Guardar todos los datos en csv
    np.savetxt(name, csv, delimiter =", ", fmt ='% s')

def main():
    opc = input('\n--- Selecciona opcion ---\n1. Generar csv para el entrenamiento\n2. Generar csv de testing\n3. Salir\n> ')
    
    if opc == '1':
        loadTrainingImages()
    elif opc == '2':
        loadTestingImages()
    elif opc == '3':
        return
    else:
        print('Opcion no contemplada')

    images.clear()
    main()

if __name__ == "__main__":
    main()


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