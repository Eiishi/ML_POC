# github : https://github.com/Eiishi/ML_POC

import pandas as pd
from joblib import load
from sklearn.metrics import silhouette_score

data = pd.read_csv("data.csv")

X = data.values

kmeans = load('kmeans.h5')

print("Score de Silhouette du K-Means seul : ", 
	silhouette_score(X, kmeans.labels_))


# Nouveau modèle
# Code inspiré : https://www.kaggle.com/code/gauravduttakiit/clustering-using-autoencoders-ann#APPLY-K-MEANS-METHOD

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

my_input = Input(shape=(5,))

x = Dense(128, activation='relu')(my_input)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

encoded = Dense(5, activation='relu')(x)

x = Dense(32, activation='relu')(encoded)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation = 'relu')(x)

decoded = Dense(5, activation = 'sigmoid')(x)

autoencoder = Model(my_input, decoded)

encoder = Model(my_input, encoded)

autoencoder.compile(optimizer= 'adam', loss='categorical_crossentropy')

autoencoder.fit(X, X, batch_size = 128, epochs = 25,  verbose = 1)

X_enc = encoder.predict(X)

ae_kmeans = kmeans.fit(X_enc)

print("Score de Silhouette du K-Means amélioré : ", 
	silhouette_score(X_enc, ae_kmeans.labels_))