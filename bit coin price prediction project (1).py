#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np 
import pandas as pd 


# In[25]:


import os
for dirname, _, filenames in os.walk('D:/bitcoin-price-prediction/bitcoin_price_1week_Test - Test.csv'):
    for filename in filenames:
        print(os.path.join(dirname, "bitcoin-price-prediction/bitcoin_price_Training"))


# In[26]:


train = pd.read_csv("D:/bitcoin_price_Training - Training.csv")
test = pd.read_csv("D:/bitcoin_price_1week_Test - Test.csv")


# In[27]:


train


# In[28]:


test


# In[29]:


train['Volume'] = train['Volume'].replace({'-' : np.nan})


# In[30]:


train['Volume'] = train['Volume'].str.replace(',', '')


# In[31]:



train['Market Cap'] = train['Market Cap'].str.replace(',', '')
test['Volume'] = test['Volume'].str.replace(',', '')
test['Market Cap'] = test['Market Cap'].str.replace(',', '')


# In[32]:


train.dropna(axis=0, inplace=True)


# In[33]:


train['Volume'] = train['Volume'].astype('int64')
train['Market Cap'] = train['Market Cap'].astype('int64')


# In[34]:


test['Volume'] = test['Volume'].astype('int64')
test['Market Cap'] = test['Market Cap'].astype('int64')


# In[35]:


train


# In[38]:


test


# In[39]:


train.dtypes, test.dtypes


# In[42]:


cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']


# In[43]:


from sklearn.preprocessing import MinMaxScaler


# In[44]:


scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.fit_transform(test[cols])


# In[45]:


train


# In[46]:


test


# In[47]:


x_train = train.drop('Market Cap', axis=1)
y_train = train['Market Cap']

x_test = test.drop('Market Cap', axis=1)
y_test = test['Market Cap']


# In[48]:


x_train.drop('Date', axis=1, inplace=True)
x_test.drop('Date', axis=1, inplace=True)


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


plt.plot(y_train)


# In[51]:


plt.plot(y_test)


# In[52]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[53]:


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor


# In[54]:


model1 = SGDRegressor()
model2 = PassiveAggressiveRegressor()
model3 = MLPRegressor()


# In[55]:


model1.fit(x_train, y_train)


# In[56]:


model2.fit(x_train, y_train)


# In[57]:


model3.fit(x_train, y_train)


# In[58]:


pred1 = model1.predict(x_test)
pred1


# In[59]:


pred2 = model2.predict(x_test)
pred2


# In[60]:


pred3 = model3.predict(x_test)
pred3


# In[61]:


from sklearn.metrics import r2_score


# In[62]:


acc1 = r2_score(y_test, pred1)
acc1*100


# In[63]:


acc2 = r2_score(y_test, pred2)
acc2*100


# In[64]:


acc3 = r2_score(y_test, pred3)
acc3*100


# In[65]:


rpred = model1.predict([[0.395119, 0.280451, 0.038213, 0.051358, 0.355668]])
rpred


# In[ ]:




