#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:52:00 2022

@author: yannis

"""

from machine_learning import *

##### KNN #####

### First Test ###

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)


predic = knn_model.predict(X_test)

score_knn = knn_model.score(X_test, y_test)

print(score_knn)

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)


## Prediction des A ##
a_index = np.where(y_test == "(A)")

y_A = y_test[a_index]
X_A = X_test[a_index]

predict_knn_A = knn_model.predict(X_A)

score_knn_A = knn_model.score(X_A, y_A)


ConfusionMatrixDisplay.from_estimator(knn_model, X_A, y_A)

"""

Première conclusion : Pour les A : ce sont à chaque fois des espaces qui sont prédits
à voir :
    
    * si c'est la même chose pour les autres lettres
    * Si on on peut optimiser l'hyperparamètre k pour avoir une meilleure prédiction
    * En dernier recours : tronquer le jeu de données pour retirer les espaces 
    qui sont la classe majoritaire. 

"""

def testModelForEachCat(model, catList, yTest, XTest, plot=False):
    """ Retourne un dictionnaire avec la valeur de prédiction du modèle 
        pour chaque catégorie + plot la matrice de prédiction if plot = True
    """
    predictions = {}
    for cat in catList:
        print(cat)
        cat_index = np.where(yTest == cat[1])
        y = yTest[cat_index]
        x = XTest[cat_index]
        predictions.update({f"{cat}": model.score(x, y)})
        
        if(plot):
            ConfusionMatrixDisplay.from_estimator(model, x,y)
    return predictions


predictions = testModelForEachCat(knn_model, categories[0:-2], y_test, X_test, plot=True)

### Optimize ###

