# Projet OpenClassrooms "Développez une Preuve de Concept"
### Sana TAYS

Le but de ce projet est de récupérer une méthode initialement mise en production et de la mettre à jour selon une méthode plus récente et basée sur de la recherche récente.

Notre étude sera basée ici sur une problématique de clustering pour le client Olist, un site e-commerce brésilien, qui souhaitait une segmentation non-supervisée de sa clientèle afin d'analyser les différents comportements d'achats.
Le jeu de données brut est disponible ici : https://www.kaggle.com/olistbr/brazilian-ecommerce
Différentes transformations y ont été appliquées pour arriver au jeu de données final, notamment la fusion des différentes tables, du feature engineering, ainsi qu'une étape de preprocessing consistant en une réduction (MinMax Scaling).

La méthode dite baseline est ici un algorithme de Kmeans entraîné sur 5 features (récence, fréquence, montant moyen du panier, review score moyen, temps de livraison moyen) et paramétré sur 4 clusters (déterminé par la méthode du coude sur le score de distorsion).
La nouvelle méthode consiste en le même modèle de Kmeans, mais nourri avec des données transformées par un auto-encoder au lieu des données brutes.

Ce répertoire contient notamment :

- le script python qui va entraîner le nouveau modèle et afficher les scores
- un fichier requirements.txt avec les dépendances à installer
- le jeu de données à utiliser
- le modèle de Kmeans initial

___

Afin de pouvoir utiliser les modèles, voici la marche à suivre :

- Téléchargez le code de ce répertoire (soit via une commande `git clone` soit en téléchargement direct)
- Ouvrez une ligne de commande et naviguez jusqu'au dossier créé
- Créez un environnement virtuel conda ou venv (optionnel mais conseillé)
- Tapez la commande `pip install requirements.txt` afin d'installer tous les modules nécessaires au fonctionnement de l'algorithme
- Tapez la commande `python app.py`

Si tout fonctionne, vous voyez s'afficher dans votre terminal les scores de performance de chacun des deux modèles et vous devriez observer que le second modèle est plus performant que le premier.

Bonne utilisation !