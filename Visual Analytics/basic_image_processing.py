#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1- Basic image processing
# by Bianka Szöllősi

# __Importing libraries and reading in images__

# In[2]:


#loading in libraries
import os
import numpy as np
import sys
import cv2
from pathlib import Path
from glob import glob
import pandas as pd


# In[3]:


#reading path to images
path_to_image = os.path.join("..", "pokemon_images")


# In[4]:


#printing each file name
for image_name in Path(path_to_image).glob("*.jpg"):
    print(image_name)


# In[5]:


#making a new image path so that the loops are easier to make
img_path = os.path.join("..", "pokemon_images", "*.jpg")


# In[6]:


#making list of image names
img_names = glob(img_path)


# __Finding the height, width and number of channels for each image__

# In[7]:


for file in img_names:
    img = cv2.imread(file)
    print(img.shape)


# Height = 120
# 
# Width = 120
# 
# Number of channels = 3

# __Splitting image into 4 equal quadrants and saving them as jpg__

# In[ ]:


#needed to install "image_slicer" package
#pip install image_slicer


# In[8]:


#importing new package
import image_slicer


# In[9]:


#creating a loop for slicing each image into 4 equal parts (the "image_slicer.slice" command automatically saves the resulting pictures)
for file in img_names:
    img = cv2.imread(file)
    image_slicer.slice(file, 4) #I realize it returns png pictures, but I couldn't fix it


# __Creating and saving a dataframe with filename, width and height for the new images__

# In[14]:


#list of the image names
image_list = [os.path.basename(image_name) for image_name in Path(path_to_image).glob("*.png")] #I realize it only works with the *.png ending because I couldn't save it as a jpg file
image_list


# In[15]:


#getting heigth and width for the new images
for image_name in Path(path_to_image).glob("*.png"): #I realize it only works with the *.png ending because I couldn't save it as a jpg file
    img = cv2.imread(file)
    print(img.shape) #seems to be the same shape as original, any idea why?


# In[21]:


#list of the image heigths
image_heigth = [img.shape[0] for image_name in Path(path_to_image).glob("*.png")] #I realize it only works with the *.png ending because I couldn't save it as a jpg file
image_heigth


# In[20]:


#list of the image heigths
image_width = [img.shape[1] for image_name in Path(path_to_image).glob("*.png")] #I realize it only works with the *.png ending because I couldn't save it as a jpg file
image_width


# In[24]:


#creating data frame
df = pd.DataFrame()
df['filename']  = image_list
df['width']  = image_width
df['heigth']  = image_heigth


# In[25]:


#viewing df
print(df)


# In[26]:


#saving df
outpath = os.path.join("..", "data", "image_df.txt")
df.to_csv(outpath, index = False)


# In[ ]:




