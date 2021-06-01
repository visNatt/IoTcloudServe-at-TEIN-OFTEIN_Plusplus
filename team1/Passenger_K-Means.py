#!/usr/bin/env python
# coding: utf-8

# # Project: Cloud-Based Machine Learing CAK part

# # Passenger

# In this project, the data of Air Traffic Passenger Statistics:
# 
# Ref. : https://data.sfgov.org/Transportation/Air-Traffic-Passenger-Statistics/rkru-6vcg
# 
# Group Students: Singgih Bekti, Shakeel Ahmed, Phing Lim
# 
# Date: Apirl, 16th 2021

# ### Necessary Libraries 

# In[49]:


import requests
import pandas as pd
from pandas.io.json import json_normalize
#from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt #ใช้ plot graph
import numpy as np
from sklearn import datasets, neighbors
import itertools
import random
from sklearn.cluster import KMeans
import csv


# ### Get Response:  
# Getting the response from API we use:

# In[55]:


response = requests.get("https://data.sfgov.org/resource/rkru-6vcg.json")


# ### See Response Code (you should get 200 OK)
# 
# 
# For more information : https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
# 
# Code: 200 --> means that response is successfull

# In[56]:


print(response.status_code)


# ### Let's See The Raw Data 

# In[57]:


print(response.json())


# To make it easy to see, we are able to see the data as table

# In[75]:


ps=pd.read_json('https://data.sfgov.org/resource/rkru-6vcg.json')
ps


# ### Testing Plot Number of passengers by Flight from Feb-Dec 2020

# In[52]:


#response = requests.get("https://data.sfgov.org/resource/rkru-6vcg.json")
#print(response.status_code)

#data given as Table, pandas should be use here!
response = pd.read_json("https://data.sfgov.org/resource/rkru-6vcg.json") #<-- Data read in pandas
pd_data = response[["activity_period","operating_airline","passenger_count"]]

my_array = pd_data.to_numpy()
#print(my_array)

dates2020 = my_array[:,0]
airline2020 = my_array[:,1] #Store data of Airline Company
num_pass2020 = my_array[:,2]

#pass_air = (airline == 'Horizon Air')
#pass_air = 1*pass_air

#print(pass_air)
#Our data can provide only 10 months:
Months2020 = np.array([202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012])
allpass2020 = []

for month2020 in Months2020:    
    import numpy as geek
    g_dates2020 = geek.ones([len(dates2020), 1], dtype = int)
    g_dates2020 = month2020*g_dates2020
    #print(g_dates)
    Dec2020 = (dates2020 == g_dates2020)
    Dec2020 = 1*Dec2020
    #print(Dec)
# Calculate number of Passengers for every months in 2020
    pass_air_dec_2020 = num_pass2020*Dec2020
    num_pass_air_2020 = np.sum(pass_air_dec_2020)
    #print(num_pass_air)
    allpass2020.append({"num_pass_air":num_pass_air_2020, "month": month2020})
# list data:
allpass2020_2 = pd.DataFrame(allpass2020)
#print(allpass2)
# X is numbers of passengers - Y is month (February --> Decmember)
X2020 = allpass2020_2.iloc[:, 0].values
y2020 = allpass2020_2.iloc[:, 1].values
Y2020 = np.linspace(2,len(y2020),len(y2020))
#print(X,Y)

plt.plot(Y2020, X2020, 'o-', color= 'g', label= "Number of Passengers")
plt.ylabel('Passengers')
plt.xlabel('Months')
plt.legend()
plt.show()


# ## K-Mean 
# 
# ### Prepare data set X : Operating Period  and Y : Number of Passengers of all Airline in July 2019 until December 2020

# We take more data from July 2019 until January 2020.

# In[53]:


# Data from July 2019 until January 2020:
response_add = pd.read_json("https://data.sfgov.org/resource/rkru-6vcg.json?$where=activity_period%20%3E%20201906")
pd_data_add = response_add[["activity_period","operating_airline","passenger_count"]]

# Array of DATA add:
my_array_add = pd_data_add.to_numpy()


# Combine DATA from July 2019 - January 2020 and Feb 2020 - Dec 2020

# In[7]:


alldata = np.concatenate([my_array_add, my_array])
df = alldata[:,(0,2)]
print(df)


# 

# In[8]:


kmean=KMeans(n_clusters=5)
label = kmean.fit(df)
label


# In[9]:


centriods = kmean.cluster_centers_
centriods
#kmean.n_iter_


# In[10]:


labels = kmean.labels_
labels


# In[11]:


y_kmean = kmean.fit_predict(df)
print(y_kmean)
len(y_kmean)


# In[12]:


#plotting the results:

for i in y_kmean:
    plt.scatter(df[y_kmean == i , 0], df[y_kmean == i , 1], label = i)
    
plt.xlabel('Activity Period')
plt.ylabel('Number of Passengers')
plt.show()


# In[13]:


all_pd_data = pd.DataFrame(alldata, columns=['Activity Period','Airline','Number Passengers'])
print(labels)
for i in labels:
    all_pd_data = all_pd_data.assign(Q = labels[i])
    
all_pd_data


# ### Saving Output

# #### write CSV output file
# since our data is the panda dataframe, we can use to_cvs to convert output to csv.

# In[14]:


compression_opts = dict(method='zip', archive_name = 'output.csv')
all_pd_data.to_csv('output.zip', index=False, compression = compression_opts)


# Previously we show the data using K-Means for making us easy to understand the data. 
# We can not see all the data one by one, but using clustering method, we can cluster the data then we can see it clearly. It is essential for helping us to decide what to do. 

# #  Color Segmentation Using K-Means
# In this case, K-Means will be used for color clustering for football. This is important for us to understand how K-Means can be used for. 

# # Color Clustering
# Let's prepare the necessary libraries

# In[15]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# Next, in this step, defining the image in RGB color space

# In[43]:


original_image = cv2.imread("C:/Users/Singgih/Downloads/football.jpg") #football.jpg
plt.imshow(original_image)


# We need to convert our image from RGB Colours Space to HSV to work ahead.
# 
# Because, according to wikipedia the R, G, and B components of an object’s color in a digital image are all correlated with the amount of light hitting the object, and therefore with each other, image descriptions in terms of those components make object discrimination difficult. Descriptions in terms of hue/lightness/chroma or hue/lightness/saturation are often more relevant.

# In[44]:


img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
plt.imshow(img)


# Next, converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.

# In[45]:


vectorized = img.reshape((-1,3))


# Then, We need to convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.

# In[46]:


vectorized = np.float32(vectorized)


# We will do cluster with k = 6
# 
# Because if you look at the image above it has 6 colors, green-colored grass and dark green, red-team, white-team, red-shadow, white-shadow.
# 
# Define criteria, number of clusters(K) and apply k-means()

# In[58]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


# OpenCV provides cv2.kmeans(samples, nclusters(K), criteria, attempts, flags) function for color clustering.
# 
# 1. samples: It should be of np.float32 data type, and each feature should be put in a single column.
# 
# 2. nclusters(K): Number of clusters required at the end
# 
# 3. criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops. 
# 
# Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:
# 
# 
# Type of termination criteria. It has 3 flags as below:
# 
# cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
# 
# cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
# 
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
# 
# 
# 4. attempts: Flag to specify the number of times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
# 
# 
# 5. flags: This flag is used to specify how initial centers are taken. Normally two flags are used for this: 
# 
# cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.

# In[67]:


K = 6
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS) 


# Now convert back into uint8.

# In[68]:


center = np.uint8(center)


# Access the labels to regenerate the clustered image

# In[69]:


res = center[label.flatten()]
result_image = res.reshape((img.shape))


# Now, compare the pictures

# In[70]:


figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()


# From that case, computer will understand about the image easily and inform us what image is that and the contain inside
