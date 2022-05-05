import cv2
import keras
import random
import tkinter as tk  
import mediapipe as mp
import common.constants as const
import common.drawing_styles as ds

'''------------
    VARIABLES 
------------'''
scores = [0, 0]
# Variables ventana
window = tk.Tk()
window.title('IAPPT MENU')
window.resizable(width=False, height=False)
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


def update_scores(jugador):
    scores[jugador] += 1

    if (scores[jugador] == 5):
        # TODO mostrar ganador y cerrar cv2
        camera.release()
        cv2.destroyAllWindows()
        

'''------------
    JUEGO
Realizar gesto de CPU y dar un ganador
------------'''
def output_winner(usr_inpt):
    cpu_inpt =  random.randrange(0,3)
    usr_winning_conditions = [usr_inpt == 0 and cpu_inpt == 2 or usr_inpt == 1 and cpu_inpt == 0 or usr_inpt == 2 and cpu_inpt == 1]
 
    if usr_inpt == cpu_inpt:
        return f'Empate (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    elif any(usr_winning_conditions):
        update_scores(1)
        return f'GANASTE (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    else:
        update_scores(0)
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
            cv2.imshow('IAPPT GAME', img) 

    camera.release()
    cv2.destroyAllWindows()


'''------------
    GUI
Funcion para crear y visualizar la ventana y sus elementos
------------'''
def GUI():
    desc = tk.Label(window, text='Bienvenido a PPT, para jugar asegurate de tener una webcam conectada,\nsi el programa no reconoce bien tus gestos acercate más a la camara.')
    opciones = tk.Label(window, text='Segun el gesto detectado se coloreará:') # Piedra
    gesto1 = tk.Label(window, text='PIEDRA', bg='#808080') # Piedra
    gesto2 = tk.Label(window, text='PAPEL', bg='#ffe5b4') # Papel
    gesto3 = tk.Label(window, text='TIJERA', bg='#ff8c00') # Tijera
    desc2 = tk.Label(window, text='Despues pulsa "espacio" para el ver resultado, gana el que llegue a 5ptos.\nPara jugar pulsa el botón:')
    jugar = tk.Button(window, text='JUGAR', command=main)

    # Posicionamiento de elementos
    desc.grid(columnspan=3, row=0, padx=10, pady=10)
    
    opciones.grid(columnspan=3, row=1, pady=(0,5))
    gesto1.grid(column=0, row=2)
    gesto2.grid(column=1, row=2)
    gesto3.grid(column=2, row=2)

    desc2.grid(columnspan=3, row=3, pady=(20,2))
    jugar.grid(columnspan=3, row=4, pady=(10,20))

    # Arrancar ventanta
    window.mainloop()   


if __name__ == '__main__':
    GUI()