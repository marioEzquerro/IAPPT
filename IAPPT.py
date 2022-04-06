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


with mp_hands.Hands(
        static_image_mode = False,
        model_complexity = 1,
        max_num_hands = 1,
        min_detection_confidence = 0.6,
        min_tracking_confidence = 0.5) as hands:

    while camara.isOpened():
        success, image = camara.read()
        if not success:
            print("Camara no detectada")
            continue

        # Convertimos la imagen de BGR opencv a RGB usado en mediapipe 
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


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

            if (gestoEsPiedra(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Piedra'
            if (gestoEsPapel(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Papel'
            if (gestoEsTijera(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Tijera'
                    
            # Mostrar la imagen vista por la camara
            mp_drawing.draw_landmarks(
                image,
                mano,
                mp_hands.HAND_CONNECTIONS,
                ds.get_hand_landmarks_style(txt),
                ds.get_hand_connections_style(txt)
            )
            cv2.imshow('IAPPT', cv2.flip(image,1))
            

        # Detener programa al pulsar 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

camara.release()
cv2.destroyAllWindows()

