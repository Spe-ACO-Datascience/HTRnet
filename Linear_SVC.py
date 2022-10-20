# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:26:38 2022

@author: manon
"""


from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def SVC_Model(x, y, nbreCV ,nb_C, C_min) : 
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=2, shuffle=True)
    
    C = [C_min*10**i for i in range(0,nb_C)]
    
    param = {'C': C,'kernel': ['rbf', 'poly', 'sigmoid']} #Hyperparamètres possibles du modèle
    

    print("---  Cross Validation : Optimisation des hyperparamètres avec GridCV --- ")
    grid = GridSearchCV(SVC(),param,refit=True,verbose=2, cv =nbreCV)
    grid.fit(X_train,y_train)
    
    print("cross validation terminée")
    print(" Le meilleur modèle est donc obtenu avec les paramètres suivants")
    print(grid.best_params_)
    
    print("Matrice de confusion pour le meilleur modèle : ")
    predictions = grid.predict(X_test)

    cm = confusion_matrix(y_test, predictions, labels=grid.classes_)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=grid.classes_)
    disp.plot()

    plt.show()
    print(" Score Accuracy pour le meilleur modèle ")
    acc= accuracy_score(y_test,predictions)
    print(acc)

    return grid


