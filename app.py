# github : https://github.com/Eiishi/ML_POC

import pandas as pd
from joblib import load
from sklearn.metrics import silhouette_score

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

data = pd.read_csv("data.csv")

X = data.values

my_input = Input(shape=(5,))

def baseline() :

	"""
	Importe le modèle initial et affiche son score de Silhouette

	Args:


	Renvoie:
		Score de performance
	"""

	kmeans = load('kmeans.h5')

	score = silhouette_score(X, kmeans.labels_)

	return score

# Nouveau modèle
# Code inspiré : https://www.kaggle.com/code/gauravduttakiit/clustering-using-autoencoders-ann#APPLY-K-MEANS-METHOD

def build_autoencoder() :

	"""
	Construit l'auto-encoder

	Args:


	Renvoie:
		La première partie du réseau et le réseau complet
	"""

	x = Dense(128, activation='relu')(my_input)
	x = Dense(64, activation='relu')(x)
	x = Dense(32, activation='relu')(x)

	encoded = Dense(5, activation='relu')(x)

	x = Dense(32, activation='relu')(encoded)
	x = Dense(64, activation='relu')(x)
	x = Dense(128, activation = 'relu')(x)

	decoded = Dense(5, activation = 'sigmoid')(x)

	return encoded, decoded

encoded, decoded = build_autoencoder()

def compile_autoencoder(decoded = decoded) :

	"""
	Compile et entraîne l'auto-encoder

	Args:
		decoded : couche de sortie de l'auto-encoder

	Renvoie:
		L'auto-encoder entraîné
	"""

	autoencoder = Model(my_input, decoded)

	autoencoder.compile(optimizer= 'adam', loss='categorical_crossentropy')

	autoencoder.fit(X, X, batch_size = 128, epochs = 25,  verbose = 1)

	return autoencoder

def encoding(X = X, encoded = encoded) :

	"""
	Encode les données initiales et entraîne le nouveau KMeans

	Args:
		X : array de données
		encoded : couche de sortie de l'encoder

	Renvoie:
		Le score de Silhouette du nouveau clustering
	"""

	encoder = Model(my_input, encoded)

	X_enc = encoder.predict(X)

	ae_kmeans = kmeans.fit(X_enc)

	score = silhouette_score(X_enc, ae_kmeans.labels_)

	return score

def main() :
	baseline()
	build_autoencoder()
	compile_autoencoder()
	encoding()

if __name__ == "__main__" :
	main()