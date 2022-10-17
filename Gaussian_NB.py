# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:08:40 2022

@author: manon
"""
from machine_learning import *
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from knn_model import testModelForEachCat

# =============================================================================
# Naive Bayes
# =============================================================================

print("Méthode du Naive Bayes")

modele_GNB = GaussianNB().fit(X_train,y_train)


predictions = modele_GNB.predict(X_test)


# =============================================================================
# Matrices de Confusions
# =============================================================================
from sklearn.metrics import ConfusionMatrixDisplay


# Matrice de confusion globale
cm = ConfusionMatrixDisplay.from_estimator(modele_GNB, X_test, y_test)

#Matrice de Confusion par lettre
pred_letter = testModelForEachCat(modele_GNB, categories[0:-2], y_test, X_test, plot=True)

# =============================================================================
# Score Accuracy
# =============================================================================

from sklearn import metrics
acc_score = metrics.accuracy_score(y_test,predictions)
print(acc_score)
# 9% accuracy score


# =============================================================================
# Cross Validation SCore
# =============================================================================

# Validation du modele

scores = cross_val_score(modele_GNB, x_reshape, y_np, cv=5)
print(scores)
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()


