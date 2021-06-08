# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:24:47 2021

@author: Md Fakrul Islam
Find Frame Distribution
"""

import pandas as pd
import pydicom
import matplotlib.pyplot as plt

from PIL import Image as im

benign_dataset_df = pd.read_csv('D:/location/benign_dataset.csv')
malignant_dataset_df = pd.read_csv('D:/location/malignant_dataset.csv')

print('benign_dataset_df shape', benign_dataset_df.shape)
print('malignant_dataset_df shape', malignant_dataset_df.shape)

#Find total frames benign

benign_image_shape= []
for index, row in benign_dataset_df.iterrows():
    #print (row["DicmPath"])
    ds = pydicom.filereader.dcmread(row["DicmPath"])
    my_array = ds.pixel_array
    no_of_images = int(my_array.shape[0])
    benign_image_shape.append(no_of_images)
    

print(benign_image_shape)
benign_dataset_df['Shape']= benign_image_shape
benign_dataset_df.to_csv('D:/location/benign_dataset_wshape.csv', index=False)



malignant_image_shape = []
for index, row in malignant_dataset_df.iterrows():
    #print (row["DicmPath"])
    ds = pydicom.filereader.dcmread(row["DicmPath"])
    my_array = ds.pixel_array
    no_of_images = int(my_array.shape[0])
    malignant_image_shape.append(no_of_images)

print(malignant_image_shape)
malignant_dataset_df['Shape']= malignant_image_shape
malignant_dataset_df.to_csv('D:/location/malignant_dataset_wshape.csv', index=False)



    