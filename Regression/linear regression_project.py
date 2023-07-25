#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt


# In[19]:


df=pd.read_csv(r'C:\Users\ganes\Desktop\Data Science\machine learnings definations\linear regression\homeprices.csv')
df


# In[20]:


df.drop(['bedrooms','age'], axis=1, inplace=True)
df


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[23]:


new_df=df.drop("price", axis=1)
new_df


# In[26]:


model=linear_model.LinearRegression()
model.fit(new_df,df.price)


# In[27]:


model.predict([[2600]])


# In[28]:


model.predict([[5000]])


# In[29]:


model.coef_


# In[30]:


model.intercept_


# In[31]:


167.30954677*5000+76692.3818707813


# In[32]:


areas_df=pd.read_csv(r'C:\Users\ganes\Desktop\Data Science\codebasics hindi\linear regression\areas.csv')
areas_df


# In[38]:


p=model.predict(areas_df)
p


# In[41]:


areas_df['prices']=p
areas_df


# In[43]:


areas_df.to_csv("prediction.csv")


# In[ ]:




