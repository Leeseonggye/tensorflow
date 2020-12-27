#!/usr/bin/env python
# coding: utf-8

# ## MNIST data 불러오기

# In[3]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from tensorflow.keras import datasets


# In[7]:


mnist = datasets.mnist


# In[27]:


(train_x,train_y),(test_x,test_y) =mnist.load_data()


# ## Image dataset 들여다보기

# In[30]:


image = train_x[0]
image.shape


# In[31]:


plt.imshow(image,'gray')
plt.show()


# ## chanel 관련
# 
# GrayScale이면 1, RGB이면 3으로 만들어줘야함
#   
#   데이터 차원수 늘리기

# In[32]:


train_x_x = np.expand_dims(train_x,-1)


# In[33]:


new_train_x = tf.expand_dims(train_x,-1)
new_train_x.shape


# In[34]:


new_train_x = train_x[...,tf.newaxis]
new_train_x.shape


# In[36]:


disp = new_train_x[1,:,:,0]
disp.shape


# In[38]:


plt.imshow(image,'gray')
plt.show()


# In[40]:


train_y.shape


# In[43]:


train_y[0],train_x[0]


# In[45]:


plt.title(train_y[0])
plt.imshow(image,'gray')
plt.show()


# ## OneHot Encoding

# In[49]:


from tensorflow.keras.utils import to_categorical


# In[51]:


to_categorical(1,5)


# In[54]:


label = train_y[0]
label_onehot = to_categorical(label, num_classes = 10)
label_onehot


# In[55]:


plt.title(label_onehot)
plt.imshow(train_x[0],'gray')
plt.show()

