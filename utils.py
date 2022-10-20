import numpy as np
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
import random
import matplotlib.pyplot as plt

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


def createDataSet(categories, troncSpace=None):

    DataDictionnary = {
        cat[1]: [el for el in selectAllImagesByCat(cat[1])] for cat in categories
    }

    if(troncSpace):
        DataDictionnary["(*) espace"] = random.choices(
            DataDictionnary["(*) espace"], k=troncSpace)

    dataset_x = []
    dataset_y = []
    for category in DataDictionnary.keys():
        for imgInfo in DataDictionnary[category]:
            path = imgInfo[0].replace('\\', '/')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (28, 28))
            dataset_x.append(resized_image)
            dataset_y.append(imgInfo[1])

    x_np = np.array(dataset_x)
    y_np = np.array(dataset_y)
    n_samples = len(x_np)
    x_reshape = x_np.reshape((n_samples, -1))
    return x_reshape, y_np


def testModelForEachCat(model, catList, yTest, XTest, title, plot=False):
    """ Retourne un dictionnaire avec la valeur de prédiction du modèle 
        pour chaque catégorie + plot la matrice de prédiction if plot = True
    """
    predictions = {}
    scores = []
    for cat in catList:
        # print(cat)
        cat_index = np.where(yTest == cat[1])
        y = yTest[cat_index]
        x = XTest[cat_index]
        score = model.score(x, y)
        predictions.update({f"{cat}": score})
        scores.append(score)
        if(plot):
            ConfusionMatrixDisplay.from_estimator(model, x, y)

    plt.bar([cat[1] for cat in catList], scores)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()
    return predictions


def plotNumberOfOccurenciesByClasses(y):
    labels = np.unique(y)
    nbre_rs = [y[y == cat].shape[0] for cat in labels]

    plt.bar(range(len(labels)), nbre_rs, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()
