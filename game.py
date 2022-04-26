import tensorflow as tf
import random
import cv2
import mediapipe as mp
import drawing_styles as ds

''' ------------
VARIABLES 
------------'''
# Preparamos una lista con los nombres de los landmarks
feature_columns = []
for label in ['muneca_Y','muneca_X','indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X','corazont_Y','corazont_X','anularp_Y','anularp_X','anulart_Y','anulart_X','meniquep_Y','meniquep_X','meniquet_Y','meniquet_X']:
    feature_columns.append(tf.feature_column.numeric_column(key=label))
GESTURES = ['Piedra','Papel','Tijera']

mp_drawing = mp.solutions.drawing_utils
# Cargar el los modelos que usaremos
mp_hands = mp.solutions.hands  
gesture_classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[32, 10],
    n_classes=3,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    model_dir='model'
)
# Entrada de video por webcam:
camera = cv2.VideoCapture(0)



''' ------------
FUNCIONES
------------'''
# Funcion para enviar los datos al modelo
def input_fn(features, batch_size=128):
    predict = {}
    for i, label in enumerate(feature_columns):
        predict[label] = [float(features[i])]

    return tf.data.Dataset.from_tensor_slices(predict).batch(batch_size)


# Funcion para determinar gesto realizado y hacer return del resultado
def model_predict(landmarks):
    predictions = gesture_classifier.predict(lambda: input_fn(landmarks))

    for pred_dict in predictions:
        return pred_dict['class_ids'][0]
         
        # probability = pred_dict['probabilities'][class_id]

        # print('Prediction is "{}" ({:.1f}%)'.format(GESTURES[class_id], 100 * probability))

# Realizar gesto de CPU y dar un ganador
def gesto_cpu_intput(usrInpt):
    cpuInpt =  random.randrange(0,2)
    if usrInpt == 0 and cpuInpt == 2:
        return 'Usuario gana!'
    elif usrInpt == 0 and cpuInpt == 1:
        return 'CPU gana'
    elif usrInpt == 1 and cpuInpt == 0:
        return 'Usuario gana!'
    elif usrInpt == 1 and cpuInpt == 2:
        return 'CPU gana', cpuInpt
    elif usrInpt == 2 and cpuInpt == 1:
        return 'Usuario gana!'
    elif usrInpt == 2 and cpuInpt == 0:
        return 'CPU gana'
    elif usrInpt == cpuInpt:
        return 'Draw'
    
# Funcion que controla la camara, encuentra y dibuja la mano y llama a model_predict con los datos recogidos
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
                accion = model_predict([
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
                gesto_cpu_intput(accion)


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
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    gesto_cpu_intput('txt')
            
            cv2.imshow('IAPPT', cv2.flip(img,1))

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
