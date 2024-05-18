import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('sonar2.csv')

# Dimensions du jeu de données
print("Dimensions du jeu de données :", data.shape)

# Aperçu des premières lignes des données
print("Aperçu des données :\n", data.head())

# Résumé statistique des caractéristiques
print("Résumé statistique :\n", data.describe())

# Répartition des données par rapport à la variable de classe
print("Répartition des données par classe :\n", data['Class'].value_counts())

# Visualisation des données avec des histogrammes
data.hist(sharex=False, sharey=False, layout=(6,10),\
        xlabelsize=1, ylabelsize=1, figsize=(12,8))
plt.show()

# correlation 
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, cmap=cm.Spectral_r, interpolation='none')
fig.colorbar(cax)
fig.set_size_inches(10,10)
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Diviser les données en variables d'entrée (X) et variable cible (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Créer un jeu de test (20% des données)
from sklearn.model_selection import train_test_split
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Mettre en place la validation croisée avec 10 folds
from sklearn.model_selection import cross_val_score

def evaluate_model(model):
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    return scores.mean()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Modèle de régression logistique
lr_model = LogisticRegression()
lr_accuracy = evaluate_model(lr_model)
print("Précision du modèle de régression logistique :", lr_accuracy)

# Modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_accuracy = evaluate_model(knn_model)
print("Précision du modèle KNN :", knn_accuracy)

# Modèle CART
cart_model = DecisionTreeClassifier()
cart_accuracy = evaluate_model(cart_model)
print("Précision du modèle CART :", cart_accuracy)

# Modèle SVM
svm_model = SVC()
svm_accuracy = evaluate_model(svm_model)
print("Précision du modèle SVM :", svm_accuracy)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model_scaled(model):
    scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')
    return scores.mean()

# Modèle de régression logistique
lr_model = LogisticRegression()
lr_accuracy_scaled = evaluate_model_scaled(lr_model)
print("Précision du modèle de régression logistique (standardisé):", lr_accuracy_scaled)

# Modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_accuracy_scaled = evaluate_model_scaled(knn_model)
print("Précision du modèle KNN (standardisé):", knn_accuracy_scaled)

# Modèle CART
cart_model = DecisionTreeClassifier()
cart_accuracy_scaled = evaluate_model_scaled(cart_model)
print("Précision du modèle CART (standardisé):", cart_accuracy_scaled)

# Modèle SVM
svm_model = SVC()
svm_accuracy_scaled = evaluate_model_scaled(svm_model)
print("Précision du modèle SVM (standardisé):", svm_accuracy_scaled)


from sklearn.model_selection import GridSearchCV

parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}

svm_model = SVC()

grid_search = GridSearchCV(svm_model, parameters, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Meilleure configuration:", grid_search.best_params_)
print("Meilleure précision:", grid_search.best_score_)



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle sur le jeu de test :", accuracy)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

y_pred_scaled = svm_model.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print("Précision du modèle SVM sur le jeu de test standardisé :", accuracy_scaled)
