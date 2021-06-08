# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:28:38 2021

@author: Md Fakrul Islam
"""

import os

IMAGE_Files_Benign = []
IMAGE_Files_Malignant = []

for x in os.listdir('D:/location/output/benign_images_mri_extended/'):
    if x.endswith(".JPG"):
        IMAGE_Files_Benign.append('D:/location/output/benign_images_mri_extended/'+x)

        
        
for x in os.listdir('D:/location/output/malignant_images_mri_extended/'):
    if x.endswith(".JPG"):
        IMAGE_Files_Malignant.append('D:/location/output/malignant_images_mri_extended/'+x)
        
        
print(len(IMAGE_Files_Benign))
print(len(IMAGE_Files_Malignant))

def countList(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
            for sub in [lst1, lst2]]


Merged_Dataset = []
Merged_Dataset = countList(IMAGE_Files_Benign, IMAGE_Files_Malignant)

Merged_Dataset[0:10]

len(Merged_Dataset)

import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import correlate1d
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import minimum_filter1d
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import sobel, generic_gradient_magnitude
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.ndimage import convolve1d
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from operator import ixor

features=['convolve1d','entropy',
'standard_deviation','Canny','prewitt','percentile_filter',
'fourier_ellipsoid','fourier_uniform','fourier_gaussian',
'generic_gradient_magnitude','minimum_filter1d','maximum_filter1d','uniform_filter1d', 
'gaussian_filter1d_x','gaussian_filter1d_y','gaussian_filter1d_z','correlate1df',
'convolve_constant_k1', 'convolve_constant_k2', 'convolve_constant_k3', 
'convolve_reflect_k1',  'convolve_reflect_k2' , 'convolve_reflect_k3',
'convolve_nearest_k1',  'convolve_nearest_k2',  'convolve_nearest_k3',
'convolve_wrap_k1',     'convolve_wrap_k2',     'convolve_wrap_k3',
'convolve_mirror_k1',   'convolve_mirror_k2',   'convolve_mirror_k3', 
'outcome']

values = []

k1 = np.array([[1,1,1],[1,1,0],[1,0,0]])
k2  = np.array([[0,1,0],[0,1,0],[0,1,0]])
k3  = np.array([[1,0,0],[0,1,0],[0,0,1]])

img = cv2.imread(Merged_Dataset[3])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

def gabor_filters(img, arr):
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []  #Create empty list to hold all kernels that we will generate in a loop
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)   
                    result = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    arr = np.append(arr, np.array(result.flatten()), axis=0)
                    num += 1 
                    #print(fimg.flatten())
    return arr

features = {}

convolve_constant_k1 = []
convolve_constant_k2 = []
convolve_constant_k3 = []

convolve_reflect_k1  = []
convolve_reflect_k2  = []
convolve_reflect_k3  = []

convolve_nearest_k1  = []
convolve_nearest_k2  = []
convolve_nearest_k3  = []

convolve_wrap_k1     = []
convolve_wrap_k2     = []
convolve_wrap_k3     = []

convolve_mirror_k1   = []
convolve_mirror_k2   = []
convolve_mirror_k3   = []

gaussian_filter1d_x  = []
gaussian_filter1d_y  = []
gaussian_filter1d_z  = []

convolve1dl          = []
entropyl             = []
standard_deviationl  = []
Cannyl               = []
prewittl             = []
percentile_filterl   = []
fourier_ellipsoidl   = []
fourier_uniforml     = []
fourier_gaussianl    = []
generic_gradient_magnitudel = []
minimum_filter1dl    = []
maximum_filter1dl    = []
uniform_filter1dl    = []
correlate1df         = []

outcome              = []

IMG_WIDTH, IMG_HEIGHT = (16, 16)

#sample = Merged_Dataset[0:5000]

X = []
Y = []


#for each_image in sample:
for each_image in Merged_Dataset:
    img = cv2.imread(each_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)  
    
    arr = np.array([])
    
    result = convolve1d(img, mode = 'nearest' ,weights=[1, 3], cval=0.75)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = entropy(img, disk(1))
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = cv2.Canny(img, 100,200)
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.prewitt(img)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.percentile_filter(img, percentile=20, size=50)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.fourier_ellipsoid(img, size=25)
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.fourier_uniform(img, size=20)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.fourier_gaussian(img, sigma=4) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = generic_gradient_magnitude(img, sobel)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
        
    result=minimum_filter1d(img, size=3)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = maximum_filter1d(img, size=3)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = uniform_filter1d(img, size=3)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result   = gaussian_filter1d(img, 1)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = gaussian_filter1d(img, 2)
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = gaussian_filter1d(img, 3)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = correlate1d(img, [-1, 1])    
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.convolve(img, k1, mode='constant', cval=0.0) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k2, mode='constant', cval=0.0)    
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k3, mode='constant', cval=0.0)    
    arr = np.append(arr, np.array(result.flatten()), axis=0)    
    
    result = ndimage.convolve(img, k1, mode='reflect',  cval=0.5) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k2, mode='reflect',  cval=0.5) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k3, mode='reflect',  cval=0.5) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)
        
    result = ndimage.convolve(img, k1, mode='nearest',  cval=1.0)
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.convolve(img, k2, mode='nearest',  cval=1.0)
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k3, mode='nearest', cval=1.0) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k1, mode='wrap', cval=1.5) 
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    result = ndimage.convolve(img, k2, mode='wrap',   cval=1.5)
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k3, mode='wrap', cval=1.5)  
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k1, mode='mirror', cval=2.0)   
    arr = np.append(arr, np.array(result.flatten()), axis=0)

    result = ndimage.convolve(img, k2, mode='mirror', cval=2.0)    
    arr = np.append(arr, np.array(result.flatten()), axis=0)    

    result = ndimage.convolve(img, k3, mode='mirror', cval=2.0)    
    arr = np.append(arr, np.array(result.flatten()), axis=0)
    
    arr = gabor_filters(img, arr)
    
    X.append(arr)
    
    if 'benign' in each_image:
        Y.append(0)
    if 'malignant' in each_image:
        Y.append(1)


print(len(X))
print(len(Y))

print(X[0])
print(len(X[0]))
print(Y[0])

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state = 150)


classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

start = time.time()
classifier.fit(X_train, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = classifier.predict(X_test)
end = time.time()
pred_time = (end - start)

#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)

print('Performence of KNeighborsClassifier')
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('fit time: ',fit_time)
print('pred_time: ',pred_time)
print('Classification Report:' ) 
print(classification_report(y_test, y_pred))
print('Confusion Matrix:' ) 
print(confusion_matrix(y_test, y_pred))
#print('AUC: ')    
#metrics.auc(fpr, tpr)    



'''
rf = RandomForestClassifier()
param = {'n_estimators': [10, 150, 300],
        'max_depth': [30, 60, 90, None]}

gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)
gs_fit = gs.fit(X_train, y_train)
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
'''

rf = RandomForestClassifier(n_estimators=90, max_depth=300, n_jobs=-1)

start = time.time()
rf_model = rf.fit(X_train, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = rf_model.predict(X_test)
end = time.time()
pred_time = (end - start)

#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)

print('Performence of RandomForestClassifier')
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('fit time: ',fit_time)
print('pred_time: ',pred_time)
print('Classification Report:' ) 
print(metrics.classification_report(y_test, y_pred))
print('Confusion Matrix:' ) 
print(confusion_matrix(y_test, y_pred))
#print('AUC: ')    
#metrics.auc(fpr, tpr)    


'''
gb = GradientBoostingClassifier()
param = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [7, 11, 15],
    'learning_rate': [0.1]
}

clf = GridSearchCV(gb, param, cv=5, n_jobs=-1)
cv_fit = clf.fit(X_train, y_train)
pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
'''

gb = GradientBoostingClassifier(n_estimators=150, max_depth=11, learning_rate=0.1)

start = time.time()
gb_model = gb.fit(X_train, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = gb_model.predict(X_test)
end = time.time()

#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)

print('Performence of GradientBoostingClassifier')
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('fit time: ',fit_time)
print('pred_time: ',pred_time)
print('Classification Report:' ) 
print(metrics.classification_report(y_test, y_pred))
print('Confusion Matrix:' ) 
print(confusion_matrix(y_test, y_pred))
#print('AUC: ')    
#metrics.auc(fpr, tpr)    


#SVM
start = time.time()
clf = svm.SVC()
clf.fit(X_train,y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = clf.predict(X_test)
end = time.time()
pred_time = (end - start)

#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)

print('Performence of SVM Classifier')
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('fit time: ',fit_time)
print('pred_time: ',pred_time)
print('Classification Report:' ) 
print(metrics.classification_report(y_test, y_pred))
print('Confusion Matrix:' ) 
print(confusion_matrix(y_test, y_pred))
#print('AUC: ')    
#metrics.auc(fpr, tpr)    

#Analytics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("model_performence.csv")
df.head() #to show top 5 rows

sns.set_style("whitegrid")
g = sns.catplot(x="Accuracy", hue="Classifier", col="Type",
                data=df, y="Precision", kind="bar", saturation=.40, aspect=.70, height=5
                );

g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
g.fig.suptitle('Model Performence-For MRI Dataset using 16*16 Images')


#GridSearch
'''
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]

svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr', probability=True), parameters, cv=5, n_jobs=-1)
svm_clf.fit(X_train, y_train)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

svm_clf.best_params_
means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print('fit time: ',fit_time)

y_pred_svm = svm_clf.predict(X_test)
print('X_test: ', X_test.shape)
metrics.accuracy_score(y_test, y_pred_svm)
print(metrics.classification_report(y_test, y_pred_svm))
print('pred_time: ',pred_time)
'''