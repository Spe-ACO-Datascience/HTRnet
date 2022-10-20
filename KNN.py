#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:02:28 2022

@author: yannis
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def KNN_Model(x, y, nbre_cv, k_max, metric):
    
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=2, shuffle=True)

    scores = []  # tableau des accuracy pour chaque k

    print("--- Cross validation pour les KNN --- ")
    for k in range(2, k_max+1):
        model = KNeighborsClassifier(n_neighbors=k)
        mean_score = np.mean(cross_val_score(model, X_train, y_train, cv=nbre_cv, scoring=metric))
        print("---")
        print(f"{metric} moyenne pour k = {k}: {mean_score}")
        print("---")
        scores.append(mean_score)

    print("cross validation terminée")

    print("--- Plot de {metric} en fonction de K ---")

    abscisse = [i for i in range(1, len(scores)+1)]

    plt.plot(abscisse, scores, label="{metric} en fonction de K")
    plt.show()

    best_accuracy = max(scores)

    best_k = scores.index(best_accuracy) + 1

    print(f"--- Le meileur modèle est obtenu pour k = {best_k} ---")

    print(f"--- Matrice de confusion pour le modèle k = {best_k} ---")

    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)

    cm = confusion_matrix(y_test, predictions, labels=best_model.classes_)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=best_model.classes_)

    plt.xticks(rotation=90)
    disp.plot()
    
    plt.show()

    return best_model
