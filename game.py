import tensorflow as tf
import random
import cv2
import mediapipe as mp
import drawing_styles as ds

FEATURE_COLUMNS = ['muneca_Y','muneca_X','indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X','corazont_Y','corazont_X','anularp_Y','anularp_X','anulart_Y','anulart_X','meniquep_Y','meniquep_X','meniquet_Y','meniquet_X']
GESTURES = ['Piedra','Papel','Tijera']

mp_drawing = mp.solutions.drawing_utils
# seleccionar el los modelos que usaremos
mp_hands = mp.solutions.hands  
model = tf.estimator.DNNClassifier(
    feature_columns=FEATURE_COLUMNS,
    hidden_units=[32, 10],
    n_classes=3,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    model_dir='model'
)

# Entrada de video por webcam:
camera = cv2.VideoCapture(0)

# Estas funciones determinan si el gesto 
# toma como parametro las posiciones las llemas de los dedos y los nudillos
# def gestoEsPiedra(muñeca ,indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
#     if ((muñeca > corzp) and (indxp < indxt) and (corzp < corzt) and (anlp < anlt) and (meñp < meñt)):
#         return True
#     elif ((muñeca < corzp) and (indxp > indxt) and (corzp > corzt) and (anlp > anlt) and (meñp > meñt)):
#         return True
#     return False

# def gestoEsPapel(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
#     if ((muñeca > corzp) and (indxp > indxt) and (corzp > corzt) and (anlp > anlt) and (meñp > meñt)):
#         return True
#     elif ((muñeca < corzp) and (indxp < indxt) and (corzp < corzt) and (anlp < anlt) and (meñp < meñt)):
#         return True
#     return False

# def gestoEsTijera(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
#     if ((muñeca > corzp) and (indxp > indxt) and (corzp > corzt) and (anlp < anlt) and (meñp < meñt)):
#         return True
#     elif ((muñeca < corzp) and (indxp < indxt) and (corzp < corzt) and (anlp > anlt) and (meñp > meñt)):
#         return True
#     return False




def gesto_cpu_intput(usrInpt):
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

            # Convertimos la imgn de BGR opencv a RGB usado en mediapipe 
            img.flags.writeable = False
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


            if results.multi_hand_landmarks:
                # usamos solo el 1º elemento encontrado ya que solo trabajaremos con una mano a la vez
                hand = results.multi_hand_landmarks[0]
                # muñeca = hand.landmark[0].y
                # indxp = hand.landmark[6].y  
                # indxt = hand.landmark[8].y  
                # corzp = hand.landmark[10].y  
                # corzt = hand.landmark[12].y  
                # anlp = hand.landmark[14].y  
                # anlt = hand.landmark[16].y  
                # meñp = hand.landmark[18].y  
                # meñt = hand.landmark[20].y  
                # txt = ''
                # global accion

                # if gestoEsPiedra(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                #     accion = 1
                # if gestoEsPapel(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                #     accion = 2
                # if gestoEsTijera(muñeca, indxp, indxt, corzp, corzt, anlp, anlt, meñp, meñt):
                #     accion = 3

                # accion = model_predict([
                #     hand.landmark[0].y, hand.landmark[0].x, 
                #     hand.landmark[6].y, hand.landmark[6].x,
                #     hand.landmark[8].y, hand.landmark[8].x,
                #     hand.landmark[10].y, hand.landmark[10].x,
                #     hand.landmark[12].y, hand.landmark[12].x,
                #     hand.landmark[14].y, hand.landmark[14].x,
                #     hand.landmark[16].y, hand.landmark[16].x,
                #     hand.landmark[18].y, hand.landmark[18].x,
                #     hand.landmark[20].y, hand.landmark[20].x
                # ])
                accion = 1
              
                

                # Mostrar la imagen vista por la camara
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
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    gesto_cpu_intput('txt')
            
            cv2.imshow('IAPPT', cv2.flip(img,1))

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
