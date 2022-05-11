# IAPPT
Trabajo de fin de grado que utiliza IA para poder jugar a piedra, papel o tijera gracias a una webcam. Usando 
los frameworks de OpenCV y TensorFlow. Para jugar ejecuta game.py.

## REQUERIMIENTOS
Para poder ejecutar los ficheros de este proyecto necesitaremos instalar mediante comandos:
-curl https://bootstrap.pypa.io/get-pip.py -o get-pippy
- python .\get-pip.py
- pip install opencv-python
- pip install ipykernel
- pip install mediapipe
- pip install tensorflow
- pip install pandas


## DESCRIPCION DE FICHEROS
- `game.py` Fichero ejectutable con el juego de PPT.
- `csv_generator` Fichero ejecutable para genererar un csv de entrenamiento/evaluacion con las imagenes en data/.
- `data` Carpeta con las imagenes y lso csv para el modelo.
- `model` Carpeta donde se almacena el modelo una vez entrenado.
- `v2_IAPPY.ipynb`Jupyter Notebook que contiene la creacion, entrenamiento y validacion del modelo.
  - `IAPPT.ipynb` (outdated).  
