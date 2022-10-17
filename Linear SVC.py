# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:26:38 2022

@author: manon
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:08:40 2022

@author: manon
"""
from machine_learning import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
from knn_model import testModelForEachCat

# =============================================================================
# Linear SVC
# =============================================================================

print("Méthode de la Classification SVC Linéaire")

model_SVC  = SVC().fit(X_train,y_train)


predictions = model_SVC.predict(X_test)


# =============================================================================
# Matrices de Confusions
# =============================================================================
from sklearn.metrics import ConfusionMatrixDisplay


# Matrice de confusion globale
cm = ConfusionMatrixDisplay.from_estimator(model_SVC, X_test, y_test)

#Matrice de Confusion par lettre
pred_letter = testModelForEachCat(model_SVC, categories[0:-2], y_test, X_test, plot=True)

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

scores = cross_val_score(model_SVC, x_reshape, y_np, cv=5)
print(scores)
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()


