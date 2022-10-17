#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:45:20 2022

@author: yannis
"""

""" 
Script principal du projet

"""

### Importations ###

#### Librairie
from sklearn.model_selection import train_test_split

#### Nos fonctions : 
    
from utils import *
from db_config import *

from KNN import KNN_Model

### Premiers test avec le data set complet ### 
""" On retire les deux dernières colonnes qui sont un doublon des espaces et l'apostrophe
    pour lequel il n'y a qu'une seule donnée. 
"""

all_categories = [el for el in selectAllCategories()][0:-2]

x_all, y_all = createDataSet(all_categories)

#### Résultats obtenus pour les différents modèles #### 

""" 
Toutes les fonctions d'entraînement des modèles font de la cross-validation et le cas 
échéant estiment les hyper-paramètres et retourne le meilleur modèle. 
"""

best_knn_model = KNN_Model(x_all, y_all, 10, 15)


#### Création des dataset train et test ####

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(x_all, y_all, test_size=0.33, random_state=2, shuffle=True)

#### Matrice de confusion pour chaque catégorie de lettre, par modèle ####

testModelForEachCat(best_knn_model, all_categories, y_test_all, X_test_all, plot=True)

""" Premières conclusions : 
    
    Pour knn Une accurarcy qui n'est pas pas très optimale est une grande confusion
    
    
"""