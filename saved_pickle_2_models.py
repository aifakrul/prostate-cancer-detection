# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:02:41 2021

@author: Md Fakrul Islam
"""

import pickle
import glob
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


def read_saved_data():
    with open('D:/location/output/x_train.pkl', 'rb') as f:        
        x_train = pickle.load(f)

    with open('D:/location/output/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('D:/location/output/y_train_one_hot.pkl', 'rb') as f:
        y_train_one_hot = pickle.load(f)

    with open('D:/location/output/y_test_one_hot.pkl', 'rb') as f:
        y_test_one_hot = pickle.load(f)
    
    with open('D:/location/output/X_for_RF.pkl', 'rb') as f:
        X_for_RF = pickle.load(f)
        
    with open('D:/location/output/test_images.pkl', 'rb') as f:
        test_images = pickle.load(f)
    
    with open('D:/location/output/test_labels.pkl', 'rb') as f:
        test_labels = pickle.load(f)
        
    with open('D:/location/output/test_labels_encoded.pkl', 'rb') as f:
        test_labels_encoded = pickle.load(f)
    
    with open('D:/location/output/train_labels_encoded.pkl', 'rb') as f:
        train_labels_encoded = pickle.load(f)
        
    with open('D:/location/output/train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)
        
    
    return x_train, x_test, y_train_one_hot, y_test_one_hot, X_for_RF, test_images, test_labels, test_labels_encoded, train_labels_encoded, train_labels    

x_train, x_test, y_train_one_hot, y_test_one_hot, X_for_RF, test_images, test_labels, test_labels_encoded, train_labels_encoded, train_labels = read_saved_data()

print(x_train.shape)
print(x_test.shape)
print(y_train_one_hot.shape)
print(y_test_one_hot.shape)
print(X_for_RF.shape)
print(test_images.shape)
print(test_labels.shape)
print(test_labels_encoded.shape)
print(train_labels_encoded.shape)
print(train_labels.shape)

SIZE =224
#############################
#Load model wothout classifier/fully connected layers
VGG16_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG16_model.layers:
	layer.trainable = False
    
VGG16_model.summary()  #Trainable parameters will be 0

from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

start = time.time()
classifier.fit(X_for_RF, train_labels_encoded)
end = time.time()
fit_time = (end - start)

#Send test data through same feature extractor process
X_test_feature  = VGG16_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

start = time.time()
#Now predict using the trained RF model. 
prediction_RF = classifier.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)
end = time.time()
pred_time = (end - start)

#Print overall accuracy
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_RF)
print(classification_report(test_labels, prediction_RF))
print(cm)
sns.heatmap(cm, annot=True)


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, max_depth=11, learning_rate=0.1)

start = time.time()
gb_model = gb.fit(X_for_RF, train_labels_encoded)
end = time.time()
fit_time = (end - start)


start = time.time()
#Now predict using the trained RF model. 
prediction_RF = gb_model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)
end = time.time()
pred_time = (end - start)

#Print overall accuracy
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_RF)
print(classification_report(test_labels, prediction_RF))
print(cm)
sns.heatmap(cm, annot=True)
