import numpy as np
import cv2
from sklearn.metrics import ConfusionMatrixDisplay

from db_config import *

def extractImageCategory(fullCat):
    extract = fullCat.split("(")
    if(len(extract) > 1):
        return extract[1].split(")")[0]
    return extract[0]


def showRequest(request):
    for el in request:
        print(el)

# print("point".split("("))

def createDataSet(categories):
    
    DataDictionnary = {
        cat[1]: [el for el in selectAllImagesByCat(cat[1])] for cat in categories
    }
    
    dataset_x = []
    dataset_y = []
    for category in DataDictionnary.keys():
        for imgInfo in DataDictionnary[category]:
            path = imgInfo[0].replace('\\', '/')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            dataset_x.append(img)
            dataset_y.append(imgInfo[1])
    
    x_np = np.array(dataset_x)
    y_np = np.array(dataset_y)
    n_samples = len(x_np)
    x_reshape = x_np.reshape((n_samples, -1))
    return x_reshape, y_np


def testModelForEachCat(model, catList, yTest, XTest, plot=False):
    """ Retourne un dictionnaire avec la valeur de prédiction du modèle 
        pour chaque catégorie + plot la matrice de prédiction if plot = True
    """
    predictions = {}
    for cat in catList:
        #print(cat)
        cat_index = np.where(yTest == cat[1])
        y = yTest[cat_index]
        x = XTest[cat_index]
        predictions.update({f"{cat}": model.score(x, y)})
        
        if(plot):
            ConfusionMatrixDisplay.from_estimator(model, x,y)
    return predictions