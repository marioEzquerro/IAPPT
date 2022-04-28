from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import common.constants as const

# feature_columns = []
# for label in const.LABELS:
#     feature_columns.append(tf.feature_column.numeric_column(key=label))

# Cargar el los modelos que usaremos
# gesture_classifier = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     hidden_units=[32, 10],
#     n_classes=3, optimizer=tf.keras.optimizers.Adam(lr=0.03),
#     model_dir='model'
# )

# def input_fn(features, batch_size=1):
#     predict = {}
#     for i, label in enumerate(feature_columns):
#         predict[label] = [float(features[i])]

#     return tf.data.Dataset.from_tensor_slices(predict).batch(batch_size)



# def model_predict(landmarks):
#     predictions = gesture_classifier.predict(lambda: input_fn(landmarks))
#     for pred_dict in predictions:
#         return pred_dict['class_ids'][0]

model = tf.saved_model.load(const.MODEL_DIR)


# inic = time.time()
# fin = time.time()

# print("--- %.3f seconds ---" % (fin - inic))





