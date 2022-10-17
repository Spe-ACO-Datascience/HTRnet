# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:50:43 2022

@author: manon
"""
from machine_learning import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from knn_model import testModelForEachCat


# =============================================================================
# Random Forest
# =============================================================================
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# =============================================================================
# Matrices de Confusions
# =============================================================================
from sklearn.metrics import ConfusionMatrixDisplay


# Matrice de confusion globale
cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

#Matrice de Confusion par lettre
pred_letter = testModelForEachCat(clf, categories[0:-2], y_test, X_test, plot=True)

# =============================================================================
# Score Accuracy
# =============================================================================

from sklearn import metrics
acc_score = metrics.accuracy_score(y_test,predictions)
print(acc_score)

# =============================================================================
# Cross Validation SCore
# =============================================================================

# Validation du modele

scores = cross_val_score(clf, x_reshape, y_np, cv=5)
print(scores)
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()


##### Naive Bayes #############

print("Méthode du Naive Bayes")
modele_GNB = GaussianNB().fit(X_train,y_train)

# Predictions
modele_GNB_predict = modele_GNB.predict(X_test)
print("accuracy sur jeu de test - régression naive bayes :")
print(metrics.accuracy_score(y_test,modele_GNB_predict))

# Matrice de confusion
cm = confusion_matrix(y_test, modele_GNB_predict) 
# Affichage graphique
# Parametrisation des axes
ax =sns.heatmap(cm, square=True, annot=True, cbar=False, fmt='g',cmap = "magma")
# indication des legendes
ax.xaxis.set_ticklabels(categories, fontsize = 12)
ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()
