#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D
import os

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_image_generator = ImageDataGenerator(rescale=1./255,
                                           vertical_flip=True,
                                           horizontal_flip=True,
                                           rotation_range=0.7,
                                           zoom_range=0.3,
                                           validation_split=0.2)

train_image_dataset = train_image_generator.flow_from_directory(r'C:\Users\Abhiram P\Downloads\VITbrainFinal\Training',
                                             batch_size=16,
                                             target_size=(64,64),
                                             class_mode='categorical',
                                             shuffle=True)
                                           


# In[3]:


test_image_generator = ImageDataGenerator(rescale=1./255,
                                         vertical_flip=True,
                                         horizontal_flip=True,
                                          zoom_range=0.3,
                                         rotation_range=0.7)

test_image_dataset = test_image_generator.flow_from_directory(r'C:\Users\Abhiram P\Downloads\VITbrainFinal\Testing',
                                                              batch_size=16,
                                                              target_size=(64,64),
                                                              class_mode='categorical',
                                                              shuffle=False)


# In[4]:


for index in range(5):
    image, label = train_image_dataset[index]
    plt.imshow(image[index])
    print(label[index])
    plt.show()


# In[5]:


for index in range(5):
    image, label = test_image_dataset[index]
    plt.imshow(image[index])
    plt.show()


# In[6]:


model1 = Sequential()

model1.add(Conv2D(16, (3,3), input_shape=(64,64,3), activation='relu'))
model1.add(MaxPooling2D((2,2)))
model1.add(Dropout(0.2))

model1.add(Conv2D(32, (3,3), activation='relu'))
model1.add(MaxPooling2D((2,2)))
model1.add(Dropout(0.2))

model1.add(BatchNormalization())
model1.add(tf.keras.layers.Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dense(4, activation='softmax'))


# In[7]:


model1.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[46]:


hist1 = model1.fit(train_image_dataset, epochs=50, validation_data=test_image_dataset)


# In[9]:


model2 = Sequential()

model2.add(Conv2D(16, (3,3), input_shape=(64,64,3), activation='relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))
model2.add(BatchNormalization())

model2.add(Conv2D(32, (3,3), activation='relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))
model2.add(BatchNormalization())

model2.add(tf.keras.layers.Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(4, activation='softmax'))


# In[10]:


model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[11]:


hist2 = model2.fit(train_image_dataset, epochs=20, validation_data=test_image_dataset)


# In[12]:


from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3, EfficientNetB3


# In[13]:


train_img_data = train_image_generator.flow_from_directory(r'C:\Users\Abhiram P\Downloads\VITbrainFinal\Training',
                                                           batch_size=16,
                                                           target_size=(224,224),
                                                           class_mode='categorical',
                                                           shuffle=True)


# In[14]:


test_img_data = test_image_generator.flow_from_directory(r'C:\Users\Abhiram P\Downloads\VITbrainFinal\Testing',
                                                           batch_size=16,
                                                           target_size=(224,224),
                                                           class_mode='categorical',
                                                           shuffle=False)


# In[15]:


resnet50 = ResNet50(include_top=False,
                    classes=4,
                    input_shape=(224,224,3),
                    pooling='avg',
                    weights='imagenet')


# In[16]:


model3 = Sequential()

model3.add(resnet50)

model3.add(tf.keras.layers.Flatten())
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))

model3.add(Dense(512, activation='relu'))
model3.add(Dense(4, activation='softmax'))


# In[17]:


model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[48]:


hist3 = model3.fit(train_img_data, epochs=10, validation_data=test_img_data)


# In[20]:


vgg16 = VGG16(include_top=False,
              classes=4,
              input_shape=(224,224,3),
              pooling='avg',
              weights='imagenet')


# In[21]:


model4 = Sequential()

model4.add(vgg16)

model4.add(tf.keras.layers.Flatten())
model4.add(Dropout(0.2))

model4.add(Dense(512, activation='relu'))
model4.add(Dense(4, activation='softmax'))


# In[22]:


model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[23]:


hist4 = model4.fit(train_img_data, epochs=20, validation_data=test_img_data)


# In[24]:


effnet = EfficientNetB3(include_top=False,
              classes=4,
              input_shape=(224,224,3),
              weights='imagenet',
              pooling='max')


# In[25]:


from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam , Adamax


# In[26]:


model6 = Sequential([
    effnet,
    BatchNormalization(axis= -1 , momentum= 0.99 , epsilon= 0.001),
    Dense(256, kernel_regularizer = regularizers.l2(l= 0.016) , activity_regularizer = regularizers.l1(0.006),
         bias_regularizer= regularizers.l1(0.006) , activation = 'relu'),
    Dropout(rate= 0.4 , seed = 75),
    Dense(4 , activation = 'softmax')
])

model6.compile(Adamax(learning_rate = 0.001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model6.summary()


# In[27]:


hist6 = model6.fit(x= train_img_data , epochs = 10 , verbose = 1 , validation_data = test_img_data ,
                   validation_steps = None , shuffle = False)


# In[28]:


inception = InceptionV3(include_top=False,
              classes=4,
              input_shape=(224,224,3),
              weights='imagenet',
              pooling='max')


# In[29]:


model5 = Sequential()

model5.add(inception)
model5.add(tf.keras.layers.Flatten())
model5.add(BatchNormalization())

model5.add(Dense(256, activation='relu'))
model5.add(Dense(4, activation='softmax'))


# In[30]:


model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[33]:


hist5 = model5.fit(train_img_data, epochs=10, validation_data=test_img_data)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(hist2_pred_, test_image_dataset.classes))


# In[36]:


import matplotlib.pyplot as plt


# In[ ]:


train_loss1 = hist1.history['loss']
val_loss1 = hist1.history['val_loss']

epochs1 = range(1, len(train_loss1) + 1)

plt.plot(epochs1, train_loss1, 'b', label='Training Loss')
plt.plot(epochs1, val_loss1, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels1 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions1 = model1.predict(test_image_dataset)
predicted_labels1 = np.argmax(predictions1, axis=1)

conf_matrix1 = confusion_matrix(true_labels1, predicted_labels1)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix1, display_labels=class_names)
disp1.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('CNN Confusion Matrix - Brain Tumor Detection')
plt.show()


# In[ ]:


train_loss2 = hist2.history['loss']
val_loss2 = hist2.history['val_loss']

epochs2 = range(1, len(train_loss2) + 1)

plt.plot(epochs2, train_loss2, 'b', label='Training Loss')
plt.plot(epochs2, val_loss2, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels2 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions2 = model2.predict(test_image_dataset)
predicted_labels2 = np.argmax(predictions2, axis=1)

conf_matrix2 = confusion_matrix(true_labels2, predicted_labels2)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2, display_labels=class_names)
disp2.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('CNN-2 Confusion Matrix - Brain Tumor Detection')
plt.show()


# In[ ]:


train_loss3 = hist3.history['loss']
val_loss3 = hist3.history['val_loss']

epochs3 = range(1, len(train_loss3) + 1)

plt.plot(epochs3, train_loss3, 'b', label='Training Loss')
plt.plot(epochs3, val_loss3, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels3 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions3 = model3.predict(test_image_dataset)
predicted_labels3 = np.argmax(predictions3, axis=1)

conf_matrix3 = confusion_matrix(true_labels3, predicted_labels3)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp3 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix3, display_labels=class_names)
disp3.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('ResNet50 Confusion Matrix - Brain Tumor Detection')
plt.show()


# In[ ]:


train_loss4 = hist4.history['loss']
val_loss4 = hist4.history['val_loss']

epochs4 = range(1, len(train_loss4) + 1)

plt.plot(epochs4, train_loss4, 'b', label='Training Loss')
plt.plot(epochs4, val_loss4, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels4 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions4 = model4.predict(test_image_dataset)
predicted_labels4 = np.argmax(predictions4, axis=1)

conf_matrix4 = confusion_matrix(true_labels4, predicted_labels4)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp4 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix4, display_labels=class_names)
disp4.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('vgg16 Confusion Matrix - Brain Tumor Detection')
plt.show()


# In[ ]:


train_loss5 = hist5.history['loss']
val_loss5 = hist5.history['val_loss']

epochs5 = range(1, len(train_loss5) + 1)

plt.plot(epochs5, train_loss5, 'b', label='Training Loss')
plt.plot(epochs5, val_loss5, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels5 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions5 = model5.predict(test_image_dataset)
predicted_labels5 = np.argmax(predictions5, axis=1)

conf_matrix5 = confusion_matrix(true_labels5, predicted_labels5)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp5 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix5, display_labels=class_names)
disp5.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('inceptionV3 Confusion Matrix - Brain Tumor Detection')
plt.show()


# In[ ]:


train_loss6 = hist6.history['loss']
val_loss6 = hist6.history['val_loss']

epochs6 = range(1, len(train_loss6) + 1)

plt.plot(epochs6, train_loss6, 'b', label='Training Loss')
plt.plot(epochs6, val_loss6, 'r', label='Validation Loss')
plt.title('Training vs. Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

true_labels6 = np.concatenate([y for x, y in test_image_dataset], axis=0)

predictions6 = model6.predict(test_image_dataset)
predicted_labels6 = np.argmax(predictions6, axis=1)

conf_matrix6 = confusion_matrix(true_labels6, predicted_labels6)

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
disp6 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix6, display_labels=class_names)
disp6.plot(cmap=plt.cm.Blues, values_format='.4g')
plt.title('EfficientNetB3 Confusion Matrix - Brain Tumor Detection')
plt.show()

