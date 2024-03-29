import os
import cv2
import csv
import mediapipe as mp
import common.constants as const

'''------------
    VARIABLES
------------'''
mp_hands = mp.solutions.hands
images = []


'''------------
    AÑADIR IMAGENES
Segun la opcion de main() se cargan las imagenes de entrenamiento o testing.
------------'''
def load_images(image_folders, path):
    for folder in image_folders:
        for file in os.listdir(folder):
            images.append(os.path.join(folder, file))
    save_to_csv(path)


'''------------
    GUARDAR LANDMAKS
Lee las imagenes de los ficheros y extrae las coordenadas para luego
guardarlas en la ubicacion especificada.
------------'''
def save_to_csv(path):
    with mp_hands.Hands(
            static_image_mode = True,
            model_complexity = 0,
            max_num_hands = 1,
            min_detection_confidence = 0.5) as hands:
        
        # Abrimos archivo para escribir en él 
        with open(path, 'w') as df:
            writer = csv.writer(df, delimiter=',', lineterminator='\n')
            # Escribimos los nombres de los campos
            labels = const.LABELS
            labels.insert(0, 'GESTO')
            writer.writerow(labels)

            for file in images:
                print(f'Convirtiendo: {file}')
                # Leer imagen y voltear horizontalmente 
                img = cv2.flip(cv2.imread(file), 1)
                # Convertir de BGR a RGB y procesar imagen
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Seleccionar que pose tiene la imagen gracias al nombre de la carpeta
                if 'rock' in file:
                    action = const.ROCK
                elif 'paper' in file:
                    action = const.PAPER
                else: 
                    action = const.SCISSORS

                if results.multi_hand_landmarks:
                    # Usamos solo el primer elemento del array ya que solo hay 1 mano
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
        
        
'''------------
    MAIN
Controla el menu del usuario y sus acciones.
------------'''
def main():
    opc = input('\n--- Selecciona opcion ---\n1. Generar csv para el entrenamiento\n2. Generar csv de testing\n3. Salir\n> ')
    
    if opc == '1':
        load_images(const.TRAINING_FOLDERS, const.TRAINING_CSV)
    elif opc == '2':
        load_images(const.TESTIING_FOLDERS, const.EVALUATION_CSV)
    elif opc == '3':
        return
    else:
        print('Opcion no contemplada')

    images.clear()
    main()

if __name__ == '__main__':
    main()
