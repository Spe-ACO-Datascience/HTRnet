# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:59:01 2022

@author: manon
"""

from machine_learning import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from knn_model import testModelForEachCat


# =============================================================================
# SGDC Classifier
# =============================================================================
SGDC  = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
SGDC.fit(X_train, y_train)
predictions = SGDC.predict(X_test)


# =============================================================================
# Matrices de Confusions
# =============================================================================
from sklearn.metrics import ConfusionMatrixDisplay


# Matrice de confusion globale
cm = ConfusionMatrixDisplay.from_estimator(SGDC, X_test, y_test)

#Matrice de Confusion par lettre
pred_letter = testModelForEachCat(SGDC, categories[0:-2], y_test, X_test, plot=True)

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

scores = cross_val_score(SGDC, x_reshape, y_np, cv=5)
print(scores)
 
import matplotlib.pyplot as plt

abscisse = [i for i in range(1, len(scores)+1)]

plt.plot(abscisse, scores)
plt.show()