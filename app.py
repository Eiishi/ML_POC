import pandas as pd
from joblib import load
from sklearn.metrics import silhouette_score

data = pd.read_csv("data.csv")

X = data.values

kmeans = load('kmeans.h5')

print("Score de Silhouette du K-Means seul : ", 
	silhouette_score(X, kmeans.labels_))

encoder = load('encoder.h5')

X_enc = encoder.predict(X)

ae_kmeans = kmeans.fit(X_enc)

print("Score de Silhouette du K-Means amélioré : ", 
	silhouette_score(X_enc, ae_kmeans.labels_))