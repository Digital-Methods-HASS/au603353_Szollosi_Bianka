#!/usr/bin/env python
# coding: utf-8

# In[2]:


#system setup
import os
import sys
sys.path.append(os.path.join(".."))

import cv2
import numpy as np

from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt


# In[ ]:


#defining whole script as main
def main():


# In[34]:


#defining image path
filename = os.path.join("..", "data", "assignment3_pic.jpg")


# In[35]:


#reading in image
image = cv2.imread(filename)


# In[30]:


#showing the picture for sanity check
jimshow(image)


# In[31]:


#printing shape of the image to be able to make a guess about where the ROI should be
image.shape


# In[36]:


#drawing a green rectangle on the image
roi_image = cv2.rectangle(image, (1400, 880), (2900, 2800), (0,255,0), 3)


# In[37]:


jimshow(roi_image, "ROI")


# In[38]:


#saving roi image
outfile = os.path.join("..", "data", "image_with_ROI.jpg")
cv2.imwrite(outfile, roi_image)


# In[39]:


#cropping original image to only roi
cropped = image[880:2800, 1400:2900]


# In[40]:


jimshow(cropped, "Cropped image")


# In[41]:


#saving cropped image
outfile = os.path.join("..", "data", "image_cropped.jpg")
cv2.imwrite(outfile, cropped)


# In[42]:


#preparing canny edge detection on cropped image
#first, making grey scale
grey_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)


# In[45]:


#defining threshold by looking at the histogram of the pixels
plt.hist(grey_image.flatten(), 256, [0,256])
plt.show()


# In[46]:


#blurring the image to prepare it for edge detection
blurred = cv2.GaussianBlur(grey_image, (5,5), 0)


# In[74]:


#applying threshold (120) by setting all values above it to white and then inverting it
(T_value, thresh_inverted) = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)


# In[75]:


jimshow_channel(thresh_inverted)


# In[1]:


#applying canny edge detection by defining min and max values manually (after playing around with it)
canny = cv2.Canny(thresh_inverted, 70, 150)


# In[78]:


#plotting canny edge detection image
jimshow_channel(canny, "Canny edge detection")


# In[68]:


#finding contours for the letters on a copied image
(contours, _) = cv2.findContours(canny.copy(),
                 cv2.RETR_EXTERNAL,
                 cv2.CHAIN_APPROX_SIMPLE)


# In[84]:


#drawing the contours on the cropped image
contoured = cv2.drawContours(cropped.copy(),
                 contours, #previously defined list of contours
                 -1, #all contours
                 (0,255,0), #green contour color
                 2) #contour pixel width


# In[83]:


#printing the contoured image for sanity check
jimshow(contoured)


# In[85]:


#saving canny image
outfile = os.path.join("..", "data", "image_letters.jpg")
cv2.imwrite(outfile, contoured)


# In[ ]:


#defining behaviour when called from command line
if __name__=="__main__":
    main()

