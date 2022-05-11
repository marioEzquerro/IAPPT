import cv2
import keras
import random
import tkinter as tk  
import mediapipe as mp
import tkinter.messagebox
import common.constants as const
import common.drawing_styles as ds
import keyboard
'''------------
    VARIABLES 
------------'''
scores = {'usr': 0, 'cpu': 0}
# Variables ventana
window = tk.Tk()
window.title('IAPPT MENU')
window.resizable(width=False, height=False)
# Cargar el los modelos que usaremos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  
gesture_classifier = keras.models.load_model("model\\gesture_classifier.h5")


'''------------
    IA
Funcion para determinar gesto realizado y hacer return del resultado.
El metodo predict devuelve una lista de resultados tipo [[% piedra, % papel, % tijera], [...]] 
aqui obtenemos el indice de la primera prediccion con mejor probabilidad.
------------'''
def model_predict(landmarks):
    if landmarks is None:
        return -1
    predictions = gesture_classifier.predict(landmarks)
    return [i for i, val in enumerate(predictions[0]) if val == max(predictions[0])][0]



'''------------
    RESTORE
Restaurar el juego al estado inicial.
------------'''
def restore_game(camera):
    scores['usr'] = 0
    scores['cpu'] = 0
    camera.release()
    cv2.destroyAllWindows()

'''------------
    SCORES
Actualizar las puntuaciones y dar ganador si 'usr' o 'cpu' llegan a 5 puntos.
------------'''
def update_scores(jugador, camera):
    scores[jugador] += 1

    # Acabar el juego y reiniciar valores
    if (scores[jugador] == const.MAX_SCORE):
        tkinter.messagebox.showinfo("IAPPT", "GANASTE JUGADOR!") if (jugador == 'usr') else tkinter.messagebox.showinfo("IAPPT", "PERDISTE")
        restore_game(camera)
       
        
'''------------
    JUEGO
Realizar gesto aleatorio para CPU y dar un ganador segun las reglas de PPT.
------------'''
def output_winner(camera, usr_inpt = -1):
    # Salir si no se recibe ningun gesto
    if usr_inpt == -1:
        return

    cpu_inpt =  random.randrange(0,3)
    usr_winning_conditions = [
        usr_inpt == const.ROCK and cpu_inpt == const.SCISSORS or 
        usr_inpt == const.PAPER and cpu_inpt == const.ROCK or
        usr_inpt == const.SCISSORS and cpu_inpt == const.PAPER
    ]
 
    if usr_inpt == cpu_inpt:
        return f'Empate (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    elif any(usr_winning_conditions):
        update_scores('usr', camera)
        return f'GANASTE (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    else:
        update_scores('cpu', camera)
        return f'cpu gana (tu:{const.GESTURES[usr_inpt]} vs cpu:{const.GESTURES[cpu_inpt]})'
    
    
'''------------
    MAIN
Funcion que controla la camara, encuentra y dibuja la mano y llama a model_predict con los datos recogidos.
------------'''
def main(camera): 
    with mp_hands.Hands(
            static_image_mode = False,
            model_complexity = 1,
            max_num_hands = 1,
            min_detection_confidence = 0.5,
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
                    # Predecir gesto con los datos
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
            

            img = cv2.flip(img,1)
            # Mostrar en imagen y texto
            cv2.putText(img, f'TU:{scores["usr"]} CPU:{scores["cpu"]}', (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)             
            cv2.imshow('IAPPT GAME', img) 

            # Mostrar resultado al pulSar 'espacio'
            wait_key = cv2.waitKey(5)
            if wait_key == 32:
                print(output_winner(camera, accion))
            # Salir al pulsar 'espacio'
            if wait_key == 27:
                restore_game(camera)
                break


'''------------
    GUI
Funcion para crear y visualizar la ventana y sus elementos.
------------'''
def GUI():
    desc = tk.Label(window, text='Bienvenido a PPT, para jugar asegurate de tener una webcam conectada,\nsi el programa no reconoce bien tus gestos acercate más a la camara.')
    opciones = tk.Label(window, text='Segun el gesto detectado se coloreará:')
    gesto1 = tk.Label(window, text='PIEDRA', bg='#808080') 
    gesto2 = tk.Label(window, text='PAPEL', bg='#ffe5b4') 
    gesto3 = tk.Label(window, text='TIJERA', bg='#ff8c00') 
    desc2 = tk.Label(window, text='Pulsa "espacio" para jugar una ronda o "esc" para salir.\nGana el que llegue a 5ptos.')
    jugar = tk.Button(window, text='JUGAR', command=lambda: main(camera = cv2.VideoCapture(0)))

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