# IAPPT
Trabajo de fin de grado que utiliza IA para poder jugar a piedra, papel o tijera gracias a una webcam. Usando 
los frameworks de OpenCV y TensorFlow. Para jugar ejecuta game.py.

### REQUERIMIENTOS
Para poder ejecutar los ficheros de este proyecto necesitaremos instalar Python 3.8.10 [Windows 10](https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe) o [Ubuntu](https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz).

Obtener pip para instalar librerias python
```shell
curl https://bootstrap.pypa.io/get-pip.py -o get-pippy
```
```shell
python get-pippy
```

Librerias utilizadas:

```shell
pip install opencv-python
```
```shell
pip install ipykernel
```
```shell
pip install mediapipe
```
```shell
pip install tensorflow
```
```shell
pip install pandas
```

### DESCRIPCION DE FICHEROS
- `game.py` Fichero ejectutable con el juego de PPT.
- `csv_generator.py` Fichero ejecutable para genererar un csv de entrenamiento/evaluacion con las imagenes en data/.
- `performance_test.py` Fichero demostrando la superioridad del modelo secuencial vs DNNClassifier.
- `data/` Carpeta con las imagenes y los csv para el modelo.
- `model/` Carpeta donde se almacena el modelo una vez entrenado.
- `v2_IAPPY.ipynb` Jupyter Notebook que contiene la creacion, entrenamiento y validacion del modelo.
  - `IAPPT.ipynb` (depricated) rendimiento de modelo pesimo.  
