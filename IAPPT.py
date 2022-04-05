import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  # estilos del formato
mp_hands = mp.solutions.hands  # modelo usado

# For webcam input:
camara = cv2.VideoCapture(0)

'''
Estas funciones determinan si el gesto 
toma como parametro las posiciones las llemas de los dedos y los nudillos
'''
def gestoEsPiedra(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((indxp < indxt) and (corzp < corzt) and (anlp < anlt) and (meñp < meñt)):
        return True
    return False

def gestoEsPapel(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((indxp > indxt) and (corzp > corzt) and (anlp > anlt) and (meñp > meñt)):
        return True
    return False

def gestoEsTijera(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
    if ((indxp > indxt) and (corzp > corzt) and (anlp < anlt) and (meñp < meñt)):
        return True
    return False


with mp_hands.Hands(
        model_complexity = 0,
        static_image_mode = True,
        max_num_hands = 1,
        min_detection_confidence = 0.5,
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
            mano = results.multi_hand_landmarks[0] # usamos solo el 1º elemento encontrado ya que solo trabajaremos con una mano a la vez
            # TODO explicar esto en doc
            indxp = mano.landmark[6].y  
            indxt = mano.landmark[8].y  
            corzp = mano.landmark[10].y  
            corzt = mano.landmark[12].y  
            anlp = mano.landmark[14].y  
            anlt = mano.landmark[16].y  
            meñp = mano.landmark[18].y  
            meñt = mano.landmark[20].y  
            txt = ''
            
            if (gestoEsPiedra(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Piedra'
            if (gestoEsPapel(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Papel'
            if (gestoEsTijera(indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt)):
                txt = 'Tijera'
              
            nueva_imagen = cv2.putText (image, txt, (200,200), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,0), 1)
            
            # Mostrar la imagen vista por la camara
            mp_drawing.draw_landmarks(
                image,
                mano,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            cv2.imshow('IAPPT', image)


        # Detener programa al pulsar 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

camara.release()
cv2.destroyAllWindows()

