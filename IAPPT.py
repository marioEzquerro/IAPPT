import random
from time import sleep
import cv2
import mediapipe as mp
import drawing_styles as ds

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  # seleccionar el modelo que usaremos

# Entrada de video por webcam:
camara = cv2.VideoCapture(0)

# Estas funciones determinan si el gesto 
# toma como parametro las posiciones las llemas de los dedos y los nudillos
def gestoEsPiedra(muñeca ,indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((muñeca > corzp) and (indxp < indxt) and (corzp < corzt) and (anlp < anlt) and (meñp < meñt)):
        return True
    elif ((muñeca < corzp) and (indxp > indxt) and (corzp > corzt) and (anlp > anlt) and (meñp > meñt)):
        return True
    return False

def gestoEsPapel(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((muñeca > corzp) and (indxp > indxt) and (corzp > corzt) and (anlp > anlt) and (meñp > meñt)):
        return True
    elif ((muñeca < corzp) and (indxp < indxt) and (corzp < corzt) and (anlp < anlt) and (meñp < meñt)):
        return True
    return False

def gestoEsTijera(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((muñeca > corzp) and (indxp > indxt) and (corzp > corzt) and (anlp < anlt) and (meñp < meñt)):
        return True
    if ((muñeca < corzp) and (indxp < indxt) and (corzp < corzt) and (anlp > anlt) and (meñp > meñt)):
        return True
    return False

def gestocpuInpt(usrInpt):
    cpuInpt =  random.randrange(1,3)
    if usrInpt == 1 and cpuInpt == 3:
        return 'Usuario gana!'
    elif usrInpt == 1 and cpuInpt == 2:
        return 'CPU gana'
    elif usrInpt == 2 and cpuInpt == 1:
        return 'Usuario gana!'
    elif usrInpt == 2 and cpuInpt == 3:
        return 'CPU gana', cpuInpt
    elif usrInpt == 3 and cpuInpt == 2:
        return 'Usuario gana!'
    elif usrInpt == 3 and cpuInpt == 1:
        return 'CPU gana'
    elif usrInpt == cpuInpt:
        return 'Draw'
    


with mp_hands.Hands(
        static_image_mode = False,
        model_complexity = 1,
        max_num_hands = 1,
        min_detection_confidence = 0.6,
        min_tracking_confidence = 0.5) as hands:

    while camara.isOpened():
        success, img = camara.read()

        if not success:
            print("Camara no detectada")
            break

        # Convertimos la imgn de BGR opencv a RGB usado en mediapipe 
        img.flags.writeable = False
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


        if results.multi_hand_landmarks:
            # usamos solo el 1º elemento encontrado ya que solo trabajaremos con una mano a la vez
            mano = results.multi_hand_landmarks[0]
            # TODO explicar esto en doc
            muñeca = mano.landmark[0].y
            indxp = mano.landmark[6].y  
            indxt = mano.landmark[8].y  
            corzp = mano.landmark[10].y  
            corzt = mano.landmark[12].y  
            anlp = mano.landmark[14].y  
            anlt = mano.landmark[16].y  
            meñp = mano.landmark[18].y  
            meñt = mano.landmark[20].y  
            txt = ''
            global accion

            if gestoEsPiedra(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                accion = 1
            if gestoEsPapel(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                accion = 2
            if gestoEsTijera(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                accion = 3

            # Mostrar la imagen vista por la camara
            mp_drawing.draw_landmarks(
                img,
                mano,
                mp_hands.HAND_CONNECTIONS,
                ds.get_hand_landmarks_style(accion),
                ds.get_hand_connections_style(accion)
            )
            
            # Detener programa al pulsar 'Esc'
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                gestocpuInpt(txt)
        
        cv2.imshow('IAPPT', cv2.flip(img,1))

camara.release()
cv2.destroyAllWindows()

