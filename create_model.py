from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

COLUMNS = ['gesto','muneca_Y','muneca_X','indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X','corazont_Y','corazont_X','anularp_Y','anularp_X','anulart_Y','anulart_X','meniquep_Y','meniquep_X','meniquet_Y','meniquet_X']
# train_path = tf.keras.utils.get_file("training_data.csv", "https://drive.google.com/file/d/12PQlDCw7lwUxyt9er0U0SqEklaQO1nAm/view?usp=sharing")
# eval_path = tf.keras.utils.get_file("eval_data.csv", "https://drive.google.com/file/d/1FhWNOS7RySzmabxKrBz0SXPdZ1uoebc_/view?usp=sharing")

dtrain = pd.read_csv('training_data.csv')
data_eval = pd.read_csv('eval_data.csv')

# eliminamos el campo con el resultado
train_result = dtrain.pop('gesto')
eval_result = data_eval.pop('gesto') 



# Especificamos los campos
feature_columns = []
for feature_name in COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(key=feature_name))

# Creacion de la funcion de input
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3
)

classifier.train(
  input_fn=lambda: input_fn(dtrain, train_result, training=True),
  steps=5000)

eval_result = classifier.evaluate(
  input_fn=lambda: input_fn(data_eval, eval_result, training=False)    )

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))