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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

#### Nos fonctions : 
    
from utils import *
from db_config import *

from KNN import KNN_Model
from RF import RandomForest_Model

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

best_knn_model = KNN_Model(x_all, y_all, 10, 15, 'accuracy')

best_random_forest_model = RandomForest_Model(x_all, y_all, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric= "accuracy")


#### Création des dataset train et test ####

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(x_all, y_all, test_size=0.33, random_state=2, shuffle=True)

#### Matrice de confusion pour chaque catégorie de lettre, par modèle ####

testModelForEachCat(best_knn_model, all_categories, y_test_all, X_test_all, plot=True)

testModelForEachCat(best_random_forest_model, all_categories, y_test_all, X_test_all, plot=True)

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

best_knn_model_cut_space = KNN_Model(x_cut_space, y_cut_space, 10, 15, 'accuracy')

best_random_forest_model_cut_space = RandomForest_Model(x_cut_space, y_cut_space, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric= "accuracy")

"""
    réduisons encore le nombre d'espace

"""
x_cut_space_300, y_cut_space_300 = createDataSet(all_categories, troncSpace=300)

x_cut_300_train, x_cut_300_test, y_cut_300_train, y_cut_300_test = train_test_split(x_cut_space_300, y_cut_space_300, test_size=0.33, random_state=2, shuffle=True)

#best_knn_model_cut_space_300 = KNN_Model(x_cut_300_train, y_cut_300_train, 10, 15, 'accuracy')

best_random_forest_model_cut_space_300 = RandomForest_Model(x_cut_300_train, y_cut_300_train, nbreTree=100, minDepth=2, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric= "accuracy")


predictions_space_300 = best_random_forest_model_cut_space_300.predict(x_cut_300_test)   

accuracy_space_300 = accuracy_score(y_cut_300_test, predictions_space_300)

print(accuracy_space_300)

cm = confusion_matrix(y_cut_300_test, predictions_space_300, labels=best_random_forest_model_cut_space_300.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=best_random_forest_model_cut_space_300.classes_)

disp.plot()

plt.show()


""" 

    Conclusion, ça ne marche pas mieux, la plupart des lettres sont toujours prédites comme des espaces
    
    Autre stratégie, essayer de faire de l'oversampling. Donc de créer de nouvelles données. 

"""

## Test de ressampling avec SMOTEENN ##

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
x_resampled, y_resampled = smote_enn.fit_resample(x_all, y_all)


best_knn_model_resample = KNN_Model(x_resampled, y_resampled, nbre_cv=10, k_max= 5, metric = 'accuracy')

best_random_forest_model_resample = RandomForest_Model(x_resampled, y_resampled, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric= "accuracy")

"""
Temps de calcul très long pour les modèles, car nombre de données beaucoup trop grand

Essayons avec un nombre d'espace réduit, pour avoir un jeu de données plus petit (environ 1000 données par classe maximum)

"""
smote_enn_cut_space = SMOTEENN(random_state=0)
x_resampled_cut_space, y_resampled_cut_space = smote_enn_cut_space.fit_resample(x_cut_space, y_cut_space)


best_knn_model_resample = KNN_Model(x_resampled_cut_space, y_resampled_cut_space, nbre_cv=10, k_max= 5, metric = 'accuracy')

best_random_forest_model_resample = RandomForest_Model(x_resampled_cut_space, y_resampled_cut_space, nbreTree=100, minDepth=2, maxDepth=4, minSplit=3, maxSplit=5, nbreCV=5, metric= "accuracy")




"""

Pour des k compris entre 1 et 5, l'accuracy est très élevée (proche de 1), à voir s'il faut augmenter le nombre de k ou pas, très bonne prédiction, mais
cela semble douteux : à développer. 

Pour les random forest, on peut constater quelque chose d'intéressant, quelque soit le nombre de split, c'est surtout la profondeur de l'arbre qui semble intéressante. 
En effet, plus la profondeur est importante, plus l'accuracy est élevée

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

rf_model = RandomForest_Model(x_resampled_cut_space, y_resampled_cut_space, nbreTree=100, minDepth=4, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric= "accuracy")


"""

Bien meilleur résultats : (attention tout de même à l'over fitting mais à chaque fois le test ne rentre pas en jeu dans la cross-validation, cf fonctions RF et KNN )
    
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
Afin de valider nos résultats, il faut regarder si le fait d'avoir fait de l'over-sampling 
sur l'ensemble du data set (train et test confondus) n'implique pas un biais pour l'accuracy. 

Par conséquent, nous allons reprendre notre data set de départ, le diviser en train et test. 
Ne faire de re-sampling que sur le train et ensuite prédire sur le test. 

Nous faison le choix de conserver une réduction des espaces au nombre de 300, pour des calculs plus rapide notamment
et parce que les résultats précédent n'étaient pas bien différents entre 1000 et 300. 

"""

X_train_not_resampled, X_test_not_resampled, y_train_not_resampled, y_test_not_resampled = train_test_split(x_cut_space, y_cut_space, test_size=0.33, random_state=2, shuffle=True)


smote_enn_test_not_resampled = SMOTEENN(random_state=0)
x_train_resampled, y_train_resampled = smote_enn_test_not_resampled.fit_resample(X_train_not_resampled, y_train_not_resampled)

## on entraine le modèle avec les données re samplé 

knn_resample = KNN_Model(x_train_resampled, y_train_resampled, nbre_cv=10, k_max= 15, metric = 'accuracy')

rf_resample =  RandomForest_Model(x_train_resampled, y_train_resampled, nbreTree=100, minDepth=4, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric= "accuracy")

predictions_not_resample = rf_resample.predict(X_test_not_resampled)   

accuracy_not_resample = accuracy_score(y_test_not_resampled, predictions_not_resample)

print(accuracy_not_resample) #0.40019474196689386


predictions_not_resample_knn = knn_resample.predict(X_test_not_resampled)   

accuracy_not_resample_knn = accuracy_score(y_test_not_resampled, predictions_not_resample_knn)

print(accuracy_not_resample_knn)

"""
Sans surprise, les résultats sont beaucoup moins bons, on a une accuracy de 40% pour le meilleur modèle
de random forest estimé avec toujours un nombre d'espace original à 1000.

pour un nombre d'espace à 300 : #0.19974874371859297
=> on reste sur 1000. 
Cependant, nous pouvons d'ores et déjà regarder quelles sont les lettres qui sont les mieux prédites

"""
## Matrice de confusion pour le meilleur modèle de RF : 

cm_best_rf = confusion_matrix(y_test_not_resampled, predictions_not_resample, labels=rf_resample.classes_)

disp_cm_best_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_best_rf, display_labels=rf_resample.classes_)

disp_cm_best_rf.plot()

plt.show()

testModelForEachCat(rf_resample, all_categories, y_test_not_resampled, X_test_not_resampled, plot=True)

## Tester le jeu de données sans espace

"""



"""

without_space = [el for el in selectAllCategories()][0:-4]

x_ws, y_ws = createDataSet(without_space)


X_ws_train, X_ws_test, y_ws_train, y_ws_test = train_test_split(x_ws, y_ws, test_size=0.33, random_state=2, shuffle=True)

rf_ws =  RandomForest_Model(X_ws_train, y_ws_train, nbreTree=100, minDepth=4, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric= "accuracy")


smote_enn_ws = SMOTEENN(sampling_strategy = "auto", random_state=0)
X_ws_train_resample, y_ws_train_resample = smote_enn_ws.fit_resample(X_ws_train, y_ws_train)

rf_ws =  RandomForest_Model(X_ws_train_resample, y_ws_train_resample, nbreTree=100, minDepth=4, maxDepth=10, minSplit=4, maxSplit=5, nbreCV=5, metric= "accuracy")

predictions_ws_not_resample = rf_ws.predict(X_ws_test)   

accuracy_ws_not_resample = accuracy_score(y_ws_test, predictions_ws_not_resample)

print(accuracy_ws_not_resample) 

plotNumberOfOccurenciesByClasses(y_ws_train_resample)
