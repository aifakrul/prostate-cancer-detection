# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:17:53 2021

@author: Md Fakrul Islam
Process Target Dataset
"""

import pandas as pd
import pydicom
from PIL import Image as im
import cv2

benign_dataset_df = pd.read_csv('D:/location/benign_dataset_wshape.csv')
malignant_dataset_df = pd.read_csv('D:/location/malignant_dataset_wshape.csv')


#Drop all duplicates Patient in Benign
benign_dataset_df.drop_duplicates(subset ="PatientID",
                     keep = False, inplace = True)

#Drop all duplicates Patient in Malignant
malignant_dataset_df.drop_duplicates(subset ="PatientID",
                     keep = False, inplace = True)


#Benign Path
dir_path_b = 'D:/location/bn_target_images/'

#Malignant Path
dir_path_m = 'D:/location/mg_target_images/'

#2% samples will be taken
FACTOR_B = 0.10

FACTOR_M = 0.01

width = 256
height = 256
dim = (width, height)


#Traverse Benign Dataset
for index, row in benign_dataset_df.iterrows():
    #print (row["DicmPath"])
    #print (row["Shape"])
    PID   = row["PatientID"]
    TPID  = PID.split('-')
    TLEN  = len(TPID)
    #print('PatiendID', TPID[TLEN-1])
    position = int((int(row["Shape"]))/2)
    limit = int(int(row["Shape"])*FACTOR_B)
    
    ds = pydicom.filereader.dcmread(row["DicmPath"])
    
    for x in range(limit):
        high = position+x
        low  = position-x
        #print('x: ', x)
        #print('Position: ', position)
        #print('High: ', high)
        #print('low: ', low)
        data = im.fromarray(ds.pixel_array[low])
        data = im.fromarray(ds.pixel_array[high])
        
        impl = dir_path_b+str(TPID[TLEN-1])+'_'+str(index)+'b_'+str(low)+''+'.png'
        imph = dir_path_b+str(TPID[TLEN-1])+'_'+str(index)+'b_'+str(high)+''+'.png'
        
        data.save(impl)
        data.save(imph)
        
        img = cv2.imread(impl)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(impl,resized)
    
        img = cv2.imread(imph)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(imph,resized)
    
#Traverse Malignant Dataset
for index, row in malignant_dataset_df.iterrows():
    #print (row["DicmPath"])
    #print (row["Shape"])
    PID   = row["PatientID"]
    TPID  = PID.split('-')
    TLEN  = len(TPID)
    #print('PatiendID', TPID[TLEN-1])
    position = int((int(row["Shape"]))/2)
    limit = int(int(row["Shape"])*FACTOR_M)
    
    ds = pydicom.filereader.dcmread(row["DicmPath"])
    
    for x in range(limit):
        high = position+x
        low  = position-x
        #print('x: ', x)
        #print('Position: ', position)
        #print('High: ', high)
        #print('low: ', low)
        data = im.fromarray(ds.pixel_array[low])
        data = im.fromarray(ds.pixel_array[high])
        
        impl = dir_path_m+str(TPID[TLEN-1])+'_'+str(index)+'m_'+str(low)+''+'.png'
        imph = dir_path_m+str(TPID[TLEN-1])+'_'+str(index)+'m_'+str(high)+''+'.png'
        
        data.save(impl)
        data.save(imph)
        
        
        img = cv2.imread(impl)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(impl,resized)
    
        img = cv2.imread(imph)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(imph,resized)
