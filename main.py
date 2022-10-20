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

# Librairie

# Nos fonctions :


### Premiers test avec le data set complet ###
""" On retire les deux dernières colonnes qui sont un doublon des espaces et l'apostrophe
    pour lequel il n'y a qu'une seule donnée. 
"""

from imblearn.combine import SMOTEENN
from Linear_SVC import SVC_Model
from sklearn.model_selection import train_test_split
from utils import *
from db_config import *
from KNN import KNN_Model
from RF import RandomForest_Model
from neural_network import NN_Model
all_categories = [el for el in selectAllCategories()][0:-2]

x_all, y_all = createDataSet(all_categories)

#### Résultats obtenus pour les différents modèles ####

""" 
Toutes les fonctions d'entraînement des modèles font de la cross-validation et le cas 
échéant estiment les hyper-paramètres et retourne le meilleur modèle. 
"""

best_knn_model = KNN_Model(x_all, y_all, 10, 15, 'accuracy')

best_random_forest_model = RandomForest_Model(
    x_all, y_all, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric="accuracy")

best_SVC_model = SVC_Model(x_all, y_all, nbreCV=5, C_min=1, nb_C=2)

#### Création des dataset train et test ####

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    x_all, y_all, test_size=0.33, random_state=2, shuffle=True)

#### Matrice de confusion pour chaque catégorie de lettre, par modèle ####

testModelForEachCat(best_knn_model, all_categories,
                    y_test_all, X_test_all, plot=True)

testModelForEachCat(best_random_forest_model, all_categories,
                    y_test_all, X_test_all, plot=True)

testModelForEachCat(best_SVC_model, all_categories,
                    y_test_all, X_test_all, plot=True)

""" Premières conclusions : 
    
    Quelque soit le modèle et quelques soit les hyper paramètres, on a la même accuracy
    Quand on regarde plus en détail, toutes les lettres sont très souvent prédites comme étant des espaces. 
    Problème de déséquilibre des classes. 
    
    Deux stratégies s'offrent à nous : 
    
        * Ré-équilibrer les données artificiellement
        * Augmenter le nombre de données (en récolter plus)
    
    Dans le cadre du projet, la dernière option qui est très certainement la meilleure, n'est pas possible. 
    Nous essaierons donc la première stratégie. 
    
    Nous avons plusieurs solutions pour réaliser cette dernière  :
        
        * Oversampling
        * undersampling
        * class weight 
        * decision threshold (ne fonctionne que pour du binaire à priori)
        
    
    Dans un premier temps, au regard de nos données, nous allons réduire manuellement le nombre d'espace. 
    C'est une décision forte, mais qui semble nécessaire au regard de la différence d'espace par rapport 
    aux autres classes. 
    
"""

### Données avec un nombre d'espace réduit ###

x_cut_space, y_cut_space = createDataSet(all_categories, troncSpace=1000)

best_knn_model_cut_space = KNN_Model(
    x_cut_space, y_cut_space, 10, 15, 'accuracy')

best_random_forest_model_cut_space = RandomForest_Model(
    x_cut_space, y_cut_space, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric="accuracy")

best_SVC_model_cut_space = SVC_Model(
    x_cut_space, y_cut_space, nbreCV=5, C_min=1, nb_C=2)

""" 

    Conclusion, ça ne marche pas mieux, la plupart des lettres sont toujours prédites comme des espaces
    
    Autre stratégie, essayer de faire de l'oversampling. Donc de créer de nouvelles données. 

"""

## Test de ressampling avec SMOTEENN ##


smote_enn = SMOTEENN(random_state=0)
x_resampled, y_resampled = smote_enn.fit_resample(x_all, y_all)


best_knn_model_resample = KNN_Model(
    x_resampled, y_resampled, nbre_cv=10, k_max=5, metric='accuracy')

best_random_forest_model_resample = RandomForest_Model(
    x_resampled, y_resampled, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric="accuracy")

best_svc_model_resample = SVC_Model(
    x_resampled, y_resampled, nbreCV=5, C_min=1, nb_C=2)


"""
Temps de calcul très long pour les modèles, car nombre de données beaucoup trop grand

Essayons avec un nombre d'espace réduit, pour avoir un jeu de données plus petit (environ 300 données maximum)

"""
smote_enn_cut_space = SMOTEENN(random_state=0)
x_resampled_cut_space, y_resampled_cut_space = smote_enn_cut_space.fit_resample(
    x_cut_space, y_cut_space)


best_knn_model_resample = KNN_Model(
    x_resampled_cut_space, y_resampled_cut_space, nbre_cv=10, k_max=5, metric='accuracy')

best_random_forest_model_resample = RandomForest_Model(
    x_resampled_cut_space, y_resampled_cut_space, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric="accuracy")


"""
Pour des k compris entre 1 et 5, l'accuracy est très élevée (proche de 1), à voir s'il faut augmenter le nombre de k ou pas, très bonne prédiction, mais
cela semble douteux : à développer. 

Pour les random forest, on peut constater quelque chose d'intéressant, quelque soit le nombre de split, c'est surtout la profondeur de l'arbre qui semble intéressante. 

--- start cross validation Random Forest --- 
---
--- accuracy moyenne pour split = 3 et depth = 2 ---
0.20501366345984712
---
---
--- accuracy moyenne pour split = 3 et depth = 3 ---
0.38028213629419916
---
---
--- accuracy moyenne pour split = 3 et depth = 4 ---
0.4770781052981096
---
---
--- accuracy moyenne pour split = 4 et depth = 2 ---
0.2202565607591816
---
---
--- accuracy moyenne pour split = 4 et depth = 3 ---
0.3735382339432027
---
---
--- accuracy moyenne pour split = 4 et depth = 4 ---
0.48043027076559064
---
---
--- accuracy moyenne pour split = 5 et depth = 2 ---
0.21296612001429982
---
---
--- accuracy moyenne pour split = 5 et depth = 3 ---
0.3669890506165368
---
---
--- accuracy moyenne pour split = 5 et depth = 4 ---
0.4792612043979891
---
--- cross validation terminée ---
--- La meilleur accuracy est 0.48043027076559064 --- 
 Pour split = 4 et depth = 4
 
 
 Un peut donc fixer un nombre de split compris entre 4 et 5 pour le moment et regarder pour une profondeur plus importante. 

"""

#### Meilleure optimisation des random forest ####

rf_model = RandomForest_Model(x_resampled_cut_space, y_resampled_cut_space, nbreTree=100,
                              minDepth=4, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric="accuracy")


"""

Bien meilleur résultats : (attention tout de même à l'over fitting mais à chaque fois le test ne rentre pas en jeu dans la fonction )
    
    --- split data set into train and test --- 
    --- start cross validation Random Forest --- 
    ---
    --- accuracy moyenne pour split = 4 et depth = 4 ---
    0.47951812082286266
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 5 ---
    0.5812297241648603
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 6 ---
    0.6821834828627249
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 7 ---
    0.7862212080390603
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 8 ---
    0.8668686400924275
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 9 ---
    0.922902530600252
    ---
    ---
    --- accuracy moyenne pour split = 4 et depth = 10 ---
    0.9521121904579107
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 4 ---
    0.4764926831390187
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 5 ---
    0.575876793671443
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 6 ---
    0.6881767434192765
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 7 ---
    0.7852903835565569
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 8 ---
    0.8649482761509718
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 9 ---
    0.9211570760178687
    ---
    ---
    --- accuracy moyenne pour split = 5 et depth = 10 ---
    0.9497264227380136
    ---
    --- cross validation terminée ---
    --- La meilleur accuracy est 0.9521121904579107 --- 
     Pour split = 4 et depth = 10

"""


"""
Deep learning method 
"""
# Sur le dataset complet

NN_Model(X_train_all, X_test_all, y_train_all, y_test_all)

"""
The validation accuracy is : [0.46915584802627563]
The training accuracy is : [0.43182727694511414]
The validation loss is : [1.8632093667984009]
The training loss is : [3.420698881149292]
Classification error:  53.08 %
"""

# Sur le dataset resampled
NN_Model(x_resampled, X_test_all, y_resampled, y_test_all)

"""
The validation accuracy is : [0.5689935088157654]
The training accuracy is : [0.1520247608423233]
The validation loss is : [1.5406628847122192]
The training loss is : [2.8868353366851807]
Classification error:  43.1 %
"""

NN_Model(x_resampled_cut_space, X_test_all, y_resampled_cut_space, y_test_all)

"""
The validation accuracy is : [0.4642857015132904]
The training accuracy is : [0.07636450976133347]
The validation loss is : [1.9020476341247559]
The training loss is : [3.236750841140747]
Classification error:  53.57 %
"""
