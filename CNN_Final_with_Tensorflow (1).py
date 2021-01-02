#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import datasets


# In[9]:


input_shape = (28,28,1)
num_classes = 10


# In[10]:


inputs = layers.Input(shape = input_shape, dtype =tf.float64)

#Feature Extraction
net = layers.Conv2D(32,(3,3),padding = 'same')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32,(3,3),padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D(pool_size=(2,2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64,(3,3),padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64,(3,3),padding = 'same')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D(pool_size = (2,2))(net)
net = layers.Dropout(0.25)(net)

#Fully Connected

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.25)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs = inputs, outputs = net, name ='Basic_CNN')


# In[11]:


mnist = datasets.mnist

(train_x,train_y),(test_x,test_y) =mnist.load_data()

train_x =train_x[...,tf.newaxis]
test_x =test_x[...,tf.newaxis]
train_x = train_x/255
test_x = test_x/255


# In[13]:


train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_ds = test_ds.batch(32)


# In[ ]:





# In[15]:


model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs = 10000)


# In[16]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# In[18]:


train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')


# In[ ]:





# In[19]:


def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables ))
    train_loss(loss)
    train_accuracy(labels, predictions)


# In[20]:


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# In[23]:


for epoch in range(2):
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
    
    print(template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100, 
                         ))


# In[ ]:





# In[25]:


model.evaluate(test_x,test_y,batch_size=32)


# In[26]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


test_image = test_x[0,:,:,0]
test_image.shape


# In[28]:


pred = model.predict(test_image.reshape(1,28,28,1))


# In[29]:


pred.shape


# In[30]:


test_batch = test_x[:32]


# In[31]:


preds = model.predict(test_batch)
preds.shape


# In[33]:


np.argmax(preds,-1)


# In[34]:


plt.imshow(test_batch[0,:,:,0],'gray')
plt.show()

