# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:41:05 2022

@author: Emilie
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from machinelearning import *

 
train_x = X_train
test_x = X_test
train_y = y_train
test_y = y_test


'''
Reshaping the data so that it can be displayed as an image
'''
train_x = np.reshape(train_x, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x, (test_x.shape[0], 28,28))
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)

word_dict = {'(A)':0, '(B)':1,'(C)':2,'(D)':3,'(E)':4,'(F)':5,'(G)':6,'(H)':7,'(I)':8,'(J)':9,'(K)':10, '(L)':11,'(M)':12,'(N)':13,'(O)':14,'(P)':15,'(Q)':16,'(R)':17,'(S)':18,'(T)':19, '(U)':20,'(V)':21,'(W)':22,'(X)':23,'(Y)':24,'(Z)':25, '(*) espace':26,'point':27}
train_y = [word_dict.get(train_y[i]) for i in range(len(train_y))]
train_y = [int(train_y[i] or 0) for i in range(len(train_y))]
test_y = [word_dict.get(test_y[i]) for i in range(len(test_y))]
test_y = [int(test_y[i] or 0) for i in range(len(test_y))]

'''
Reshaiping data so it can fit the model
'''
train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("New shape of train data: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of train data: ", test_X.shape)


train_yOHE = to_categorical(train_y, num_classes = 28, dtype='int')
print("New shape of train labels: ", train_yOHE.shape)

test_yOHE = to_categorical(test_y, num_classes = 28, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)


'''
Creation of the model 
'''
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(28,activation ="softmax"))


'''
Compiling and fitting model 
'''
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X,test_yOHE))


model.summary()
model.save(r'model_hand.h5')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])


'''
Doing Some Predictions on Test Data
'''
# Representation on pictures 
fig, axes = plt.subplots(1010, figsize=(8,9))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(test_X[i], (28,28))
    ax.imshow(img, cmap="Greys")
    
    pred = list(word_dict.keys())[np.argmax(test_yOHE[i])]    
    ax.set_title("Prediction: "+pred)
    ax.grid()

# Get a sentence for each sentence  
sentence = []
for i in range(len(test_X)):
    img = np.reshape(test_X[i], (28,28))
    
    pred = list(word_dict.keys())[np.argmax(test_yOHE[i])] 
    letter = str(pred)
    if len(letter)==3:
        letter = letter[1]
    elif len(letter) == 10:
        letter = " "
    else :
        letter = "."
    number = int(i)
    sentence.append(letter)
# Affiche la phase, même si ca n'a pas vraiment de sens pour l'instant
# faire un dataset test avec les images organisées 
print("".join(sentence)) 


''' 
Visualisatio of intermediate representations
'''
from keras.models import Model
img = test_X[12]
img = img.reshape(1,train_x.shape[1],train_x.shape[2],1)

layer_name = 'conv2d_26'
intermediate_layer_model = Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(img)
import matplotlib.pyplot as plt
import numpy as np ## to reshape
%matplotlib inline
temp = intermediate_output.reshape(28,28,1) # 2 feature
plt.imshow(temp[:,:,2],cmap='gray') 
# note that output should be reshape in 3 dimension



from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([img])[0] ## pass as input image

'''
Validation on the test set
'''
# Pas encore sûre de cette partie 
from sklearn.metrics import confusion_matrix
y_proba = model.predict(test_X)
C = []
for i in range(len(y_proba)):
    #print(i)
    C.append(list(word_dict.keys())[np.argmax(y_proba[i])] )
print(C)
M = confusion_matrix(y_test,C)
print("Confusion matrix")
print(M)
print("Classification error: ", np.round((1-np.sum(np.diag(M))/np.shape(test_X)[0])*100,2),"%")
