#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:13:35 2022

@author: yannis

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def RandomForest_Model(x, y, nbreTree, minDepth, maxDepth, minSplit, maxSplit, nbreCV, metric):
    
    scores = []
    splits = []
    depths = []
    
    print("--- start cross validation Random Forest --- ")
    
    for split in range(minSplit, maxSplit+1):
        for depth in range(minDepth, maxDepth+1):
            model = RandomForestClassifier(max_depth=depth, min_samples_split=split)
            mean_score = np.mean(cross_val_score(model, x, y, cv=nbreCV, scoring=metric))
            print("---")
            print(f"--- {metric} moyenne pour split = {split} et depth = {depth} ---")
            print(mean_score)
            print("---")
            
            splits.append(split)
            depths.append(depth)
            scores.append(mean_score)
            
    print("--- cross validation terminée ---")
    
    best_accuracy = max(scores)
    best_accuracy_index = scores.index(best_accuracy)
    
    best_split = splits[best_accuracy_index]
    best_depth = depths[best_accuracy_index]
    
    print(f"--- La meilleur {metric} est {best_accuracy} --- ")
    print(f" Pour split = {best_split} et depth = {best_depth}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=2, shuffle=True)
    
    best_model = RandomForestClassifier(max_depth=best_depth, min_samples_split=best_split)
    
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)

    cm = confusion_matrix(y_test, predictions, labels=best_model.classes_)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=best_model.classes_)

    disp.plot()

    plt.show()

    return best_model