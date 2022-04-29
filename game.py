import random
import cv2
import mediapipe as mp
import time
import drawing_styles as ds
import keras
import common.constants as const

'''------------
    VARIABLES 
------------'''
# Cargar el los modelos que usaremos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  
gesture_classifier = keras.models.load_model('model\gesture_classifier.h5')
# Almacenar entrada de video de webcam:
camera = cv2.VideoCapture(0)


'''------------
    IA
Funcion para determinar gesto realizado y hacer return del resultado
el metodo predict devuelve una lista del tipo [[x, y, z]] siendo cada valor
------------'''
def model_predict(landmarks):
    prediction = gesture_classifier.predict(landmarks)
    # return [i for i, val in enumerate(prediction[0]) if val == max(prediction[0])][0]
    for i, val in enumerate(prediction[0]):
        if val == max(prediction[0]):
            print('Indice enct', val ,i)

    
    

'''------------
    JUEGO
Realizar gesto de CPU y dar un ganador
------------'''
def output_winner(usrInpt):
    cpuInpt =  random.randrange(0,3)
    if usrInpt == 0 and cpuInpt == 2:
        return f'Usuario gana! ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})'
    elif usrInpt == 0 and cpuInpt == 1:
        return f'CPU gana ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})'
    elif usrInpt == 1 and cpuInpt == 0:
        return f'Usuario gana! ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})'
    elif usrInpt == 1 and cpuInpt == 2:
        return f'CPU gana ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})'
    elif usrInpt == 2 and cpuInpt == 1:
        return f'Usuario gana! ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})' 
    elif usrInpt == 2 and cpuInpt == 0:
        return f'CPU gana ({const.GESTURES[usrInpt]} vs {const.GESTURES[cpuInpt]})'
    elif usrInpt == cpuInpt:
        return 'Draw'
    
'''------------
    MAIN
Funcion que controla la camara, encuentra y dibuja la mano y llama a model_predict con los datos recogidos
------------'''
import time
def main():
    with mp_hands.Hands(
            static_image_mode = False,
            model_complexity = 1,
            max_num_hands = 1,
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.5) as hands:

        while camera.isOpened():
            success, img = camera.read()

            if not success:
                print("Camara no detectada")
                continue

            # Convertimos la imgn de BGR opencv a RGB usado en mediapipe 
            img.flags.writeable = False
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:      
                    accion = model_predict([[
                        hand.landmark[0].y, hand.landmark[0].x, 
                        hand.landmark[6].y, hand.landmark[6].x,
                        hand.landmark[8].y, hand.landmark[8].x,
                        hand.landmark[10].y, hand.landmark[10].x,
                        hand.landmark[12].y, hand.landmark[12].x,
                        hand.landmark[14].y, hand.landmark[14].x,
                        hand.landmark[16].y, hand.landmark[16].x,
                        hand.landmark[18].y, hand.landmark[18].x,
                        hand.landmark[20].y, hand.landmark[20].x
                    ]])             

                    # Estilizar los landmarks
                    mp_drawing.draw_landmarks(
                        img,
                        hand,
                        mp_hands.HAND_CONNECTIONS,
                        ds.get_hand_landmarks_style(accion),
                        ds.get_hand_connections_style(accion)
                    )
                
            # Detener programa al pulsar 'Esc'
            if cv2.waitKey(5) & 0xFF == 27:
                break
            elif 0xFF == 32:
                print(output_winner(accion))

            # Mostrar en imagen
            cv2.imshow('IAPPT', cv2.flip(img,1))

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
