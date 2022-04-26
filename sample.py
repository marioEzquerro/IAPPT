import tensorflow as tf
import pandas as pd

LABELS = ['muneca_Y','muneca_X','indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X','corazont_Y','corazont_X','anularp_Y','anularp_X','anulart_Y','anulart_X','meniquep_Y','meniquep_X','meniquet_Y','meniquet_X']
GESTURES = ['Piedra','Papel','Tijera']

feature_columns = []
for key in LABELS:
    feature_columns.append(tf.feature_column.numeric_column(key=key))

print(type(feature_columns[2]))

model = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[32, 10],
    n_classes=3,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    model_dir='model'
)

def input_fn(features, batch_size=128):
    return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)


def model_predict(landmarks):
    predic = {}
    for i, coord in enumerate(feature_columns):
        predic[coord] = [float(landmarks[i])]


    predictions = model.predict(lambda: input_fn(predic))

    # para borrar V
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(GESTURES[class_id], 100 * probability))
        # return class_id


model_predict([0.7521454691886902, 0.7284806966781616, 0.5133205652236938, 0.5722395181655884, 0.42854106426239014, 0.5254275798797607, 0.46435123682022095, 0.6363590359687805, 0.3522628843784332, 0.6168778538703918, 0.5148862600326538, 0.6875309348106384, 0.6417578458786011, 0.6958463788032532, 0.5651730298995972, 0.7264944911003113, 0.6593970060348511, 0.7194741368293762])