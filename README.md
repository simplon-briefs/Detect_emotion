# Detect_emotion

Lien du dataset[ ici](https://drive.google.com/file/d/1VOnamgj5pOwQL7FM3Hww6p03O-9EfQVy/view?usp=sharing) !

## Introduction 
---
Un détecteur des émotions permet de détecter et d’analyser les émotions capturées à un instant t à partir d’une simple photo où une vidéo. Ce service peut identifier jusqu’à 7 émotions : la colère, le dégoût, la peur, le bonheur, neutre, la tristesse et la surprise. Vous allez construire un modèle IA qui permet de réaliser cette tâche sur des images et des vidéos.

## Dependance 
---
- Python3
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
- [Tensorflow](https://www.tensorflow.org)
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

## Installation
---
- Premier, git clone et rentré dans le dossier.

```shell
git clone https://github.com/simplon-briefs/Detect_emotion.git
cd Detect_emotion
```

- Second, unzip le model.zip
```shell
unzip model.h5.zip -d .
```
- Troisieme, lancer sur jupyter-notebook le fichier app-detect_emotion.ipynb

```shell
jupyter-notebook
```
selectionner **app-detect_emotion.ipynb**
puis lancer le programme 

## Entrainement du model
---
- Premier, telecharger les données [ ici](https://drive.google.com/file/d/1VOnamgj5pOwQL7FM3Hww6p03O-9EfQVy/view?usp=sharing) 
<br><br>
- Seconde mettre le fichier **RN-detect_emotion.ipynb** et les donnée sur [google drive](https://www.google.com/intl/fr_tg/drive/)
<br><br>
- Troisieme sur lancer le fichier **RN-detect_emotion.ipynb**
sur google colab <br>
```text
Clique droit sur le fichier
Ouvrir avec
Google colab
```
- Quatrième, activer le GPU
```text
Edit 
Notebook settings
Selectioner GPU
```
- Cinquième lancer le nootebook
```text
Runtime
run all
```
---
## Reseau de neurone 
```python
batch_size = 128
num_epoch = 60


model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_categories, activation='softmax'))
```
