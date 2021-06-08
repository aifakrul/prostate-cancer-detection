# -*- coding: utf-8 -*-
"""
Created on Wed May 19 23:56:08 2021

@author: Md Fakrul Islam
"""

import pydicom
import matplotlib.pyplot as plt

from PIL import Image as im

import cv2

path ='D:/location'
#file_path = path + '/x.dcm'
#file_path = path + '/1_mri.dcm'

file_path = 'D:/location/mri_samples_mg/1-01.dcm'
print(file_path)

dir_path = 'D:/location/mri_images/'

ds = pydicom.filereader.dcmread(file_path)
print(ds)

print()
print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print(f"Patient's Name...: {display_name}")
print(f"Patient ID.......: {ds.PatientID}")
print(f"Modality.........: {ds.Modality}")
print(f"Study Date.......: {ds.StudyDate}")
print(f"Image size.......: {ds.Rows} x {ds.Columns}")
print(f"Pixel Spacing....: {ds.PixelSpacing}")

# use .get() if not sure the item exists, and want a default value if missing
print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

my_array = ds.pixel_array
print('Shape is: ', my_array.shape)
print('Series Instance ID: ', ds.SeriesInstanceUID)


no_of_images = int(my_array.shape[0])
print('No of Images: ', no_of_images)

print('Type: ',type(ds.pixel_array[226]))

#data = im.fromarray(ds.pixel_array[226])
#data.save('gfg_dummy_pic.png')

width = 224
height = 224
dim = (width, height)

for i in range(0, no_of_images):
    data = im.fromarray(ds.pixel_array[i])
    #data.save(dir_path+str(i)+'_'+ds.PatientID+'.png')
    impath= dir_path+str(i)+''+'.png'
    data.save(impath)
    img = cv2.imread(impath)
    #print('Original Dimensions : ',img.shape)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',resized.shape)
    cv2.imwrite(impath,resized) 
 
    #print('I: ',i)

# plot the image using matplotlib
#plt.imshow(ds.pixel_array[113], cmap=plt.cm.gray)
#plt.imshow(ds.pixel_array[0], cmap=plt.cm.gray)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()