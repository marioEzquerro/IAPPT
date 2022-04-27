import time
import tensorflow as tf
import common.constants as const

feature_columns = []
for label in const.LABELS:
    feature_columns.append(tf.feature_column.numeric_column(key=label))
    
# Cargar el los modelos que usaremos
gesture_classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[32, 10],
    n_classes=3,optimizer=tf.keras.optimizers.Adam(lr=0.03),
    model_dir='model'
)


def input_fn(features, batch_size=1):
    predict = {}
    for i, label in enumerate(feature_columns):
        predict[label] = [float(features[i])]

    return tf.data.Dataset.from_tensor_slices(predict).batch(batch_size)

# Funcion para determinar gesto realizado y hacer return del resultado
def model_predict(landmarks):
    predictions = gesture_classifier.predict(lambda: input_fn(landmarks))    
    for pred_dict in predictions:        
        return pred_dict['class_ids'][0] 

inic = time.time()

# model_predict([0.6964344382286072,0.4340957999229431,0.29846206307411194,0.6515616178512573,0.10236793756484985,0.7110217213630676,0.24927093088626862,0.4965308904647827,0.024313032627105713,0.478731632232666,0.2766636610031128,0.4066409468650818,0.07552826404571533,0.43055054545402527,0.341887503862381,0.2891741394996643,0.18166279792785645,0.27778539061546326])

fin = time.time()

model = tf.Module(gesture_classifier)
model.

gesture_classifier.export_saved_model(export_dir_base='a', serving_input_receiver_fn=)
print("--- %.3f seconds ---" % (fin - inic))

