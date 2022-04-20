import csv
import os
import cv2
import mediapipe as mp
from keras.preprocessing.image import ImageDataGenerator


mp_hands = mp.solutions.hands
TRAINING_FOLDERS = ['.\\data\\train\\paper', '.\\data\\train\\scissors', '.\\data\\train\\rock']
TESTIING_FOLDERS = ['.\\data\\test\\paper', '.\\data\\test\\scissors', '.\\data\\test\\rock']
ROCK = 1
PAPER = 2
SCISSORS = 3

images = []

# Creamos un objeto del tipo ImageDataGenerator para transformar imagenes
# datagen = ImageDataGenerator(
#     rotation_range=180,
#     horizontal_flip=True,
#     fill_mode='nearest')

# Añadir todas las imagenes a la lista
def loadTrainingImages():
    for folder in TRAINING_FOLDERS:
        for file in os.listdir(folder):
                images.append(os.path.join(folder, file))
    saveData('training_data.csv')
    
def loadTestingImages():
    for folder in TESTIING_FOLDERS:
        for file in os.listdir(folder):
            images.append(os.path.join(folder, file))
    saveData('eval_data.csv')
    
# leer landmarks y guardarlas
def saveData(name):
    print('Generando csv...')   
    with mp_hands.Hands(
            static_image_mode = True,
            model_complexity = 0,
            max_num_hands = 1,
            min_detection_confidence = 0.5) as hands:
        
        # abrimos archivo para escribir en él 
        with open(name, 'w') as df:
            writer = csv.writer(df, delimiter=',', lineterminator='\n')
            # escribimos los nombres de los campos
            writer.writerow(['gesto', 'indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X', 'corazont_Y','corazont_X', 'anularp_Y','anularp_X', 'anulart_Y','anulart_X', 'meniquep_Y','meniquep_X','meniquet_Y','meniquet_X'])

            for idx, file in enumerate(images):
                print(f'Convirtiendo: {file}')
                # Leer imagen y voltear horizontalmente 
                img = cv2.flip(cv2.imread(file), 1)
                # Convertir de BGR a RGB antes de procesar
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # determinar que imagen se esta procesando via el path en el que esta guardado
                if 'rock' in file:
                    action = ROCK
                elif 'paper' in file:
                    action = PAPER
                else: 
                    action = SCISSORS

                if results.multi_hand_landmarks:
                    # Usamos solo el primer elemento del array que contiene las manos encontradas
                    hand = results.multi_hand_landmarks[0]
                    
                    # Escribir landmarks de la mano
                    writer.writerow([action,
                        hand.landmark[0].y, hand.landmark[0].x, 
                        hand.landmark[6].y, hand.landmark[6].x,
                        hand.landmark[8].y, hand.landmark[8].x,
                        hand.landmark[10].y, hand.landmark[10].x,
                        hand.landmark[12].y, hand.landmark[12].x,
                        hand.landmark[14].y, hand.landmark[14].x,
                        hand.landmark[16].y, hand.landmark[16].x,
                        hand.landmark[18].y, hand.landmark[18].x,
                        hand.landmark[20].y, hand.landmark[20].x
                    ])
        
       
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