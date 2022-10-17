from db_config import *
from utils import *
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
# images = selectImages(10)
# for i in images:
#     img = cv2.imread(i[0])
#     cv2.imshow(f'{extractImageCategory(i[1])}', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

### PLOT DATA Informations ###

# Get All categories
categories = [el for el in selectAllCategories()]

# chaque catégorie est une liste de tupple
allDataInformations = {
    cat[1]: [el for el in selectAllImagesByCat(cat[1])] for cat in categories
}
# print(len(allDataInformations["(A)"]))

names = list(allDataInformations.keys())
values = list(allDataInformations.values())

nbre = [len(el) for el in values]

## All values

plt.bar(range(len(allDataInformations)), nbre, tick_label=names)
plt.xticks(rotation=90)
plt.show()


## Letters only

plt.bar(range(0,26), nbre[0:26], tick_label=names[0:26])
plt.xticks(rotation=90)
plt.show()


# Load Images

listValues = [v for v in values]

## Exemple de code pour charger la premièr image (forcément un A)
for listOfVall in listValues:
    for val in listOfVall:        
        img = cv2.imread(val[0], cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap="gray")
        plt.show()
        break
    break



### CREATE DATASET ###

## A FAIRE ## 

""" 
    Dans la fonction de création du dataset : il faudrait mettre le même nombre
    de données pour chaque catégorie de lettre => prendre le nombre minimal et 
    tronqué les autres tableaux de données de manière aléatoire
    
    Sinon sur représentation de l'espace, des E et des A 
    il semble y avoir un doublon pour les espaces
    
    
    Il faut vérifier si les labels sont ok en character ou s'il faut du numérique
    auquel cas il faudra prendre les index de chaque catégorie. 
"""

def createDataSet(DataDictionnary):
    dataset_x = []
    dataset_y = []
    for category in DataDictionnary.keys():
        for imgInfo in DataDictionnary[category]:
            path = imgInfo[0].replace('\\', '/')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img , (28, 28))
            dataset_x.append(resized_image)
            dataset_y.append(imgInfo[1])
    return dataset_x, dataset_y


x, y = createDataSet(allDataInformations)

x_np = np.array(x)
y_np = np.array(y)

#show an image of dataset
plt.imshow(x[567], cmap="gray")
plt.show()

## A FAIRE ## 

""" 
    Mtn qu'on a le dataset, il faut faire un train et un test avec un split 
    (voir sklearn pour ça)
    
"""

## Train et split du dataset 
# il faut reshape les images : 
n_samples = len(x_np)
x_reshape = x_np.reshape((n_samples, -1))


X_train, X_test, y_train, y_test = train_test_split(x_reshape, y_np, test_size=0.33, random_state=2, shuffle=True)


## Test avec SVM (support vector classifier)


    

# from sklearn.svm import SVC
# svm_model = SVC(gamma=0.001)

# svm_model.fit(X_train, y_train)

# predict = svm_model.predict(X_test)

##### KNN #####

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)


predic = knn_model.predict(X_test)


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)


