import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image
from sharpening import unsharp_mask
from hog import hogProcess
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, recall_score, precision_score, f1_score


from statistics import *
from ELMR import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer

def preprocess_img(image):
    image = cv2.resize(image, (64, 64))
    # Sharpen image
    image = unsharp_mask(image)
    # Convert to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoise image
    image = ndimage.gaussian_filter(image, 3)
    # Binarize image
    ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # Get HOG features
    image = hogProcess(image)
    # Convert image to array
    image = np.array(image).flatten()
    return image

def BrailleImage(imaged):
    pick_in = open('braille_model.pickle','rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)
    model = ELMClassifier(n_hidden=500, activation_func='multiquadric')
    model.fit(xtrain, ytrain)

    pick = open('braille_ELM.sav','rb')
    model=pickle.load(pick)
    pick.close()

    #imgPath = imageDir # sample "Braille2C_Datasets\\karaniwan\\karaniwan_28.jpg"
    image = cv2.imread(imaged,1)
    image = preprocess_img(image)
    image = np.expand_dims(image, 0)
    print(image.shape)
    ##print(image.shape)
    prediction = model.predict(xtest)
    accuracy = model.score(xtest,ytest)

    categories = ['binata', 'buhay', 'dalaga', 'eksamen', 'ewan', 'gunita', 'halaman', 'hapon', 'isip', 'kailangan',
                 'karaniwan', 'kislap', 'larawan', 'mabuti', 'noon', 'opo', 'papaano', 'patuloy', 'roon', 'subalit',
                  'talaga', 'ugali', 'wasto']

    print('Accuracy of the model: ', accuracy)
    print('Recall xtest: ', recall_score(ytest, prediction, pos_label='positive', average='macro'))
    print('Precision xtest: ', recall_score(ytest, prediction, pos_label='positive', average='macro'))

    if accuracy*100 > 70:

        #print(confusion_matrix(ytest,prediction))
        #print(classification_report(ytest,prediction))

        prediction1 = model.predict(image)
        recall_sc = recall_score(ytest,prediction1, pos_label='positive', average='macro')
        #precision_sc = precision_score(ytest,prediction1, pos_label='positive', average='macro')
        #mse = np.mean(np.square(ytest - prediction1)+50)
        #r2sc=r2_score(ytest,prediction1)

        print('Prediction is: ', categories[prediction1[0]])
        print('Recall: ', recall_sc)
        print('Precision: ', precision_sc)

    else:
        print('Image is not a Two-cell Braille!')

    return prediction1, accuracy, recall_sc, precision_sc


pred, acc, recall_scc, precision_scc =BrailleImage('G:\\PythThesis\\PyProject\\venv\\ewan_1.jpg')

print('Scores: ',pred,' ',acc,' ',recall_scc,' ',precision_scc)