
# coding: utf-8

# In[173]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[174]:

def gini(solution, submission):
    df = zip(solution, submission)
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    solution=np.array(solution)
    submission=np.array(submission)
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini


# In[175]:

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[176]:

train_df.head()


# In[179]:

train_df.describe()


# In[178]:

# Check if there's any NaN values in the dataframe
train_df.isnull().values.sum()


# In[177]:

labels = train_df.Hazard


# In[180]:

insurance_X_train = train_df.ix[:,["T1_V%d" % i for i in [1,2,3,10,13,14]] + ["T2_V%d" % j for j in [1,2,4,6,7,8,9,10,14,15]]].values
insurance_Y_train = train_df.Hazard
insurance_X_tets = test_df.ix[:,["T1_V%d" % i for i in [1,2,3,10,13,14]] + ["T2_V%d" % j for j in [1,2,4,6,7,8,9,10,14,15]]].values


# In[181]:

train_x, test_x, train_y, test_y = train_test_split(insurance_X_train, insurance_Y_train, test_size=.2)


# In[182]:

clf = linear_model.LinearRegression()
clf.fit(train_x, train_y)


# In[188]:

predictor_test = clf.predict(test_x)
predictor_train = clf.predict(train_x)


# In[189]:

print predictor_x[:5]
print predictor_y[:5]


# In[192]:

print normalized_gini(train_y, predictor_train)
print normalized_gini(test_y, predictor_test)


# In[ ]:



