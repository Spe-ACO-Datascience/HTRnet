from db_config import *
from utils import *
import cv2
import matplotlib.pyplot as plt
import os
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

def createDataSet(DataDictionnary):
    dataset = []
    for category in DataDictionnary.keys():
        for imgInfo in DataDictionnary[category]:
            path = imgInfo[0].replace('\\', '/')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            dataset.append([img, imgInfo[1]])
    return dataset


dataset = createDataSet(allDataInformations)

#show an image of dataset
plt.imshow(dataset[245][0], cmap="gray")
plt.show()


