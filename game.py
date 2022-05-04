import random
import cv2
import mediapipe as mp
import common.drawing_styles as ds
import common.constants as const
import keras

'''------------
    VARIABLES 
------------'''
scores = [0, 0]
# Cargar el los modelos que usaremos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  
gesture_classifier = keras.models.load_model("model\\gesture_classifier.h5")
# Almacenar entrada de video de webcam:
camera = cv2.VideoCapture(0)


'''------------
    IA
Funcion para determinar gesto realizado y hacer return del resultado.
El metodo predict devuelve una lista de resultados tipo [[% piedra, % papel, % tijera]] 
aqui obtenemos el indice de la primera prediccion con mejor probabilidad.
------------'''
def model_predict(landmarks):
    predictions = gesture_classifier.predict(landmarks)
    return [i for i, val in enumerate(predictions[0]) if val == max(predictions[0])][0]


'''------------
    JUEGO
Realizar gesto de CPU y dar un ganador
------------'''
def output_winner(usr_inpt):
    cpu_inpt =  random.randrange(0,3)
    usr_winning_conditions = [usr_inpt == 0 and cpu_inpt == 2 or usr_inpt == 1 and cpu_inpt ==   0 or usr_inpt == 2 and cpu_inpt == 1]
 
    if usr_inpt == cpu_inpt:
        return f'Empate (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    elif any(usr_winning_conditions):
        scores[1] += 1
        return f'GANASTE (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    else:
        scores[0] += 1
        return f'cpu gana (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    
    
'''------------
    MAIN
Funcion que controla la camara, encuentra y dibuja la mano y llama a model_predict con los datos recogidos
------------'''
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
                break

            # Convertimos la imgen BGR de opencv a RGB para procesarla usado en mediapipe 
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

            # Detener al pulsar 'Esc' o mostrar ganador 'space'
            # if cv2.waitKey(5) & 0xFF == 27:                                            
            #     break   
            if cv2.waitKey(5) & 0xFF == ord(' '):
                print(output_winner(accion))

            img = cv2.flip(img,1)
            cv2.putText(img, f'CPU:{scores[0]} TU:{scores[1]}', (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)             
            # Mostrar en imagen
            cv2.imshow('PPT GAME', img) 

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
