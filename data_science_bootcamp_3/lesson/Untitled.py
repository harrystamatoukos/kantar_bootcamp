#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# ### Learning objectives
# - Define simple linear regression mathematically
# - build a linear regression model
# 

# data/bikeshare.csv

# In[1]:


#import the necessary libraries
import pandas as pd


# In[2]:


#load the bikeshare data
bikes = pd.read_csv('data/bikeshare.csv', index_col='datetime', parse_dates = True)


# In[3]:


pwd


# In[4]:


#View my dataset


# In[5]:


bikes.head()


# ## What is that something that we are trying to predict ?  

# In[6]:


#A: the count, which is the number of bikes we are renting


# In[7]:


#rename count to 'total_rentals'


# In[8]:


bikes = bikes.rename(columns={'count':'total_rentals'})
#bikes.rename(columns={'count':'total_rentals'}, inplace = True)


# In[9]:


bikes


# ## Visualise the data

# In[10]:


import matplotlib.pyplot as plt


# In[11]:


bikes.plot(kind='scatter', x='temp', y='total_rentals')


# ## Build a linear regression model using scikit learn 

# - the predicted value and the features should be different objects
# - all of the values should be numeric
# - the should be a form of numpy array or pandas series, or dataframe
# 

# In[12]:


#create a datafrane with the temperature column
feature_cols = ['temp']
X = bikes[feature_cols]  #pandas dataframe


# In[13]:


# create a series with the total_rentals
y = bikes['total_rentals']  # pandas series


# In[14]:


get_ipython().run_cell_magic('time', '', "feature_cols = ['temp']\nX = bikes[feature_cols]")


# In[15]:


get_ipython().run_cell_magic('time', '', "x = pd.DataFrame(bikes, columns= ['temp'])")


# #### Step1 : iimport the library and the model that we need 

# In[16]:


from sklearn.linear_model import LinearRegression


# #### Step2: Instantiate the estimator

# - Make an instance of something
# - estimator is the sklearn object or the LinearRegression

# In[17]:


lr  = LinearRegression()


# In[18]:


lr2 = LinearRegression()


# In[19]:


type(lr)


# - Created an object that knows how to do linear regression
# - how we name it doesn't matter

# #### Step 3: Fit the model with data 

# In[20]:


lr.fit(X, y)


# - Once a model has been trained it is called fitted model

# #### Step 4: Make predictions with our model

# In[21]:


predictions = lr.predict(X)


# In[22]:


predictions


# In[23]:


#creating a new column
bikes['predictions'] = predictions


# In[24]:


bikes[['temp','predictions']]


# In[25]:


bikes.plot(kind='scatter', x = 'temp',y= 'total_rentals')
plt.plot(bikes.temp, bikes.predictions)


# In[26]:


print(lr.intercept_)


# In[27]:


print(lr.coef_)


# In[28]:


lr.predict([[1]])


# In[29]:


lr.predict([[2]])


# # Does the Scale of the Features Matter 

# In[30]:


bikes['temp_F'] = bikes.temp * 1.8 + 32


# In[31]:


# Recreate the linear regression with Fahrenheit instead of Celcius 


# In[32]:


feature_cols = bikes['temp_F']


# In[33]:


type(bikes['temp'])


# In[34]:


###


# # Working with Multiple Features

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[36]:


bikes


# In[37]:


feature_cols = ['temp',  'season','weather','humidity']


# In[38]:


# relationships among feature cols and total rentals 
sns.pairplot(data = bikes, x_vars = feature_cols, y_vars = 'total_rentals', kind='scatter')


# In[39]:


# Let's look at rentals over time


# In[40]:


bikes.total_rentals.plot(figsize=(16,6), style='.')


# In[41]:


# he wave an overal growth 


# In[42]:


bikes.corr()


# In[43]:


sns.heatmap(bikes.corr(), annot = True)


# In[44]:


plt.rcParams['figure.figsize'] = (16,5)


# In[45]:


feature_cols


# #### Create another linear model but this time instead of one X, we have many 

# In[46]:


from sklearn.linear_model import LinearRegression

# Create X and y
X = bikes[feature_cols]
y = bikes['total_rentals']

# Instantiate your model 
lr = LinearRegression()

# fit your model
lr.fit(X, y)          

#print intercept and coeffients
print(lr.intercept_)
print(lr.coef_)


# In[47]:


list(zip(feature_cols, lr.coef_))


# # Multicollinearity

# In[48]:


feature_cols = ['temp', 'atemp']


# In[49]:


X = bikes[feature_cols]
y = bikes['total_rentals']


# In[50]:


linreg = LinearRegression()


# In[51]:


linreg.fit(X,y)


# In[52]:


print(linreg.coef_)


# In[53]:


list(zip(feature_cols, linreg.coef_))


# In[54]:


linreg.intercept_


# # How to do Feature Selection

# In[55]:


predictions = [8, 6, 5, 10]
actual = [10, 7, 5 ,5]

import numpy as np


# In[56]:


from sklearn import metrics

# MAE : Mean Absolute Error
print(metrics.mean_absolute_error(actual, predictions))

# MSE : Mean Squared Error
print(metrics.mean_squared_error(actual, predictions))
# RMSE: Root MEan Squared Error
print(np.sqrt(metrics.mean_squared_error(actual, predictions)))


# # Train/Test split

# In[57]:


bikes


# In[58]:


#import train_test_split library
from sklearn.model_selection import train_test_split


# In[59]:


feature_cols = ['temp','humidity','weather','season']


# In[60]:


X = bikes[feature_cols]
y = bikes['total_rentals']


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)


# In[ ]:





# In[62]:



def train_test_rmse( df, feature_cols):
    
    ''' A function that takes as input a dataset and a 
    list of features and returns the root mean squared error of the predictions'''
    
    #Create X, y
    X = df[feature_cols]
    y = df['total_rentals']
    
    #Split the data in training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 123)
    
    #instanitate a linear regression model
    lr = LinearRegression()
    
    # fit a model on the training data
    lr.fit(X_train, y_train)
    
    #Create predictions based on testing data
    preds = lr.predict(X_test)
    
    # return the error
    return np.sqrt(metrics.mean_squared_error(y_test, preds))


# In[63]:


train_test_rmse(bikes, ['temp','humidity'])


# In[64]:


train_test_rmse(bikes, ['temp','humidity','weather'])


# In[65]:


train_test_rmse(bikes, ['temp','humidity','weather','season'])


# In[ ]:





# In[66]:


X = bikes[feature_cols]
y = bikes['total_rentals']


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)


# In[68]:


#instanitate a linear regression model
lr = LinearRegression()
    
# fit a model on the training data
lr.fit(X_train, y_train)
    
#Create predictions based on testing data
preds = lr.predict(X_test)


# In[69]:


X_test


# In[70]:


preds


# In[85]:


get_ipython().system('jupyter nbconvert --to script Untitled.ipynb')


# In[ ]:




