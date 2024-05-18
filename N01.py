import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le jeu de données dans un DataFrame
data = pd.read_csv('iris.csv')
# Dimensions du jeu de données
print("Dimensions du jeu de données :", data.shape)
# Aperçu des premières lignes des données
print("Aperçu des données :\n", data.head())
# Résumé statistique des caractéristiques
print("Résumé statistique :\n", data.describe())
# Répartition des données par rapport à la variable de classe
print("Répartition des données par classe :\n", data['species'].value_counts())
# Visualisation des données avec des histogrammes
data.hist()
plt.show()
# Visualisation des données avec des plots
sns.pairplot(data, hue='species')
plt.show()

#Evaluation des algorithmes demanadés
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# charger Iris directement
iris = load_iris()

# Séparer les entités (X) et les étiquettes cibles (y)
X = iris.data
y = iris.target

# Diviser les données en ensembles d’entraînement et de test (70 % d’entraînement, 30 % de test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Définir différents modèles de classification
models = []
models.append(("La Refgression Logistique", LogisticRegression(multi_class="ovr", solver="lbfgs")))
models.append(("KNN", KNeighborsClassifier(n_neighbors=5)))
models.append(("L'arbre de decision", DecisionTreeClassifier()))
models.append(("SVM", SVC(kernel="linear")))

# Évaluez chaque modèle et imprimez les résultats
for name, model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"{name} Précision: {accuracy:.4f}")

# Sélectionnez le modèle le plus performant en fonction de sa précision
MM, meilleur_model = models[0]
Meilleur_score = accuracy_score(y_test, models[0][1].predict(X_test))
for name, model in models[1:]:
  accuracy = accuracy_score(y_test, model.predict(X_test))
  if accuracy > Meilleur_score:
    MM, meilleur_model = name, model
    Meilleur_score = accuracy

print(f"\nMeilleur modèle: {MM} avec comme précision: {Meilleur_score:.4f}")
