
# coding: utf-8

# In[48]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


# In[30]:

df = pd.read_csv("train.csv")


# In[46]:

df.head()


# In[35]:

print "Number of rows: {:d}, Number of Columns: {:d}".format(df.shape[0], df.shape[1])


# In[43]:

# Check if there's any NaN values in the dataframe
df.isnull().values.sum()


# In[47]:

df.describe()


# In[92]:

# df.Hazard.values
# [getattr(df, 'T1_V%d' % i).values for i in [1,2,3,10,13,14]]
insurance_X_train = df.ix[:,["T1_V%d" % i for i in [1,2,3,10,13,14]] + ["T2_V%d" % j for j in [1,2,4,6,7,8,9,10,14,15]]].values
insurance_Y_train = df.Hazard.values


# In[93]:

clf = linear_model.LinearRegression()
clf.fit(insurance_X_train, insurance_Y_train)


# In[ ]:



