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

from sklearn.model_selection import cross_val_score

scores = []

for k in range(1,11):
    print(k)
    model = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(model, x_reshape, y_np, cv=5)))
    
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()

""" Conclusion de ce premier essai avec un nombre d'espace tronqué: 
    Les prédictions ne sont pas bonnes dans tous les cas, est-ce qu'il y a trop de classe
    par rapport au nombre de données ? 
    
    Test suivant : ne faire les tests que sur un certain nombre de lettre

"""

onlyFiveLettersDictionnary = {
    cat[1]: [el for el in selectAllImagesByCat(cat[1])] for cat in categories[0:4]
    }

x_five, y_five = createDataSet(onlyFiveLettersDictionnary)


x_five_np = np.array(x_five)
y_five_np = np.array(y_five)

n_five_samples = len(x_five_np)
x_five_reshape = x_five_np.reshape((n_five_samples, -1))

scores = []

#model_five = KNeighborsClassifier(n_neighbors=3)
#scores_five = cross_val_score(model, x_five_reshape, y_five_np, cv=2)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_five_reshape, y_five_np, test_size=0.33, random_state=2, shuffle=True)

for k in range(1,11):
    print(k)
    model = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(model, x_five_reshape, y_five_np, cv=5)))
    
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()

# si on prend 3 voisins 

knn_3_five = KNeighborsClassifier(n_neighbors=2)

knn_3_five.fit(X_train_f, y_train_f)


ConfusionMatrixDisplay.from_estimator(knn_3_five, X_test_f, y_test_f)