#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from tensorflow.keras import datasets


# In[5]:


mnist = datasets.mnist


# In[51]:


(train_x,train_y),(test_x,test_y) =mnist.load_data()


# In[58]:


image = train_x[0]
image.shape


# In[59]:


image = image[tf.newaxis,...,tf.newaxis]


# In[60]:


image.shape


# In[61]:


train_x_x = np.expand_dims(train_x,-1)


# In[62]:


tf.keras.layers.Conv2D(filters=3,kernel_size=(3,3),strides=(1,1),padding ='same', activation ='relu')


# In[63]:


tf.keras.layers.Conv2D(3,3,1,'same')


# ## Visualization

# In[64]:


image = tf.cast(image,dtype = tf.float32)
image.dtype


# In[73]:


layer = tf.keras.layers.Conv2D(3,3,1,'same')
layer


# In[74]:


output = layer(image)


# In[76]:


output


# In[78]:


plt.imshow(output[0,:,:,0],'gray')
plt.show()


# In[80]:


weight = layer.get_weights()


# In[82]:


len(weight)


# In[88]:


plt.figure(figsize=(15,5))
plt.subplot(131)
plt.hist(output.numpy().ravel(),range=[-2,2])
plt.ylim(0,100)
plt.subplot(132)
plt.title(weight[0].shape)
plt.imshow(weight[0][:,:,0,0],'gray')
plt.subplot(133)
plt.title(output.shape)
plt.imshow(output[0,:,:,0],'gray')
plt.colorbar()
plt.show()


# In[89]:


act_layer = tf.keras.layers.ReLU()
act_output = act_layer(output)


# In[101]:


pool_layer = tf.keras.layers.MaxPool2D(pool_size =(2,2),strides = (2,2),padding ='same')
pool_output = pool_layer(act_output)
pool_output.shape


# ## Fully connected
# ### Flatten

# In[103]:


tf.keras.layers.Flatten()


# In[105]:


layer = tf.keras.layers.Flatten()


# In[107]:


flatten = layer(output)


# In[109]:


output.shape


# In[111]:


flatten.shape


# ## Dense

# In[113]:


tf.keras.layers.Dense(32,activation = 'relu')


# In[115]:


layer = tf.keras.layers.Dense(32,activation = 'relu')
output = layer(flatten)


# In[117]:


output.shape


# In[125]:


layer_2 =  tf.keras.layers.Dense(64,activation = 'relu')
output_ex = layer_2(output)


# In[127]:


output_ex.shape


# ## Dropout

# In[129]:


layer = tf.keras.layers.Dropout(0.7)
output = layer(output)


# In[131]:


output.shape


# In[133]:


from tensorflow.keras import layers


# In[135]:


input_shape = (28,28,1)
num_classes = 10


# In[139]:


inputs = layers.Input(shape = input_shape)

#Feature Extraction
net = layers.Conv2D(32,3,padding = 'same')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32,3,padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2,2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64,3,padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64,3,padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2,2))(net)
net = layers.Dropout(0.25)(net)

#Fully Connected

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.25)(net)
net = layers.Dense(10)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs = inputs, outputs = net, name ='Basic_CNN')



# In[141]:


model


# In[143]:


model.summary()

