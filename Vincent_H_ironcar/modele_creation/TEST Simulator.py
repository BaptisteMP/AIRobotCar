
# coding: utf-8

# In[1]:


from roadsimulator.simulator import Simulator


# In[2]:


import os
import numpy as np

from tqdm import tqdm
from scipy.misc import imread


# In[3]:


simulator = Simulator()
#Simulator class


# In[4]:


from roadsimulator.colors import White, DarkShadow
from roadsimulator.layers.layers import Background, DrawLines, Perspective, Crop


# In[5]:


dark = DarkShadow()
#on décide de tracer une ligne noire, on peut importer les autres couleurs (White,yellow...) et modifier cela dans DrawLines

simulator.add(Background(n_backgrounds=3, path='./test_ground', input_size=(250, 200)))
simulator.add(DrawLines(input_size=(250, 200), color_range=dark))
simulator.add(Perspective())
simulator.add(Crop())


# In[6]:


simulator.generate(n_examples=2000, path='my_dataset')
#generate n_examples road pictures in the folder my_dataset


# In[7]:


from roadsimulator.models.utils import get_datasets
#associate pictures and their labels, and create the training, validating and testing sets.
train_X, train_Y, val_X, val_Y, test_X, test_Y = get_datasets('my_dataset', n_images=3000)


# In[8]:


from keras.layers import Input, Convolution2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dense
from keras.models import Model


# In[9]:


#Create the model (convolutive and dense layers)

img_in = Input(shape=(190, 250, 3), name='img_in')
x = img_in

x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

flat = Flatten()(x)

x = Dense(20)(flat)
x = Activation('relu')(x)

x = Dense(5)(x)
angle_out = Activation('softmax')(x)

model = Model(input=[img_in], output=[angle_out])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([train_X], train_Y, batch_size=20, nb_epoch=20, validation_data=([val_X], val_Y))


# In[10]:


#upload new pictures stored in 'pat' to test the model predictions on it (after this function, they are ready to be analyzed by
# model.predict() )

def selectedTest(pat):
    
    if isinstance(pat, str):
        paths = [pat]
    
    images = []

    for path in paths:
        print(path)
        for image_file in tqdm(os.listdir(path)):
            
            print(os.listdir(path))
            if '.jpg' not in image_file: continue
            try:
                img = imread(os.path.join(path, image_file))
                print(os.path.join(path, image_file))
                if img is not None:
                    images.append(img[:, :])
                    
            except Exception as e:
                pass

    images = np.array(images)
    
    return images.astype('float32') / 255.


# In[23]:


#a new set of pictures from 'selectTest' folder
imagesTest = selectedTest('selectTest')


# In[10]:


#exact label accuracy in the testing Set (10% of the total pictures)
sum([np.argmax(model.predict(test_X[i:i+1]))==np.argmax(test_Y[i]) for i in range(len(test_X))])/len(test_X)


# In[11]:


#label equal +- 1 (direction) accuracy in the testing Set (10% of the total pictures)
sum([int(abs(np.argmax(model.predict(test_X[i:i+1]))-np.argmax(test_Y[i]))<=1) for i in range(len(test_X))])/len(test_X)


# In[21]:


len(test_X)


# In[31]:


#prediction of the pictures in 'selectTest' and the angle predicted, that we can compare to the corresponding picture in the 
#folder, to see if the model predicts well, or discover bias.
#In roadSimulator, in utils, is defined the discretization of the angle, here it is -inf -> -0.75 / -0.75 -> -0.25 / -0.25 ->
#0.25 / 0.25 -> 0.75 / 0.75 -> +inf; creating 5 categories in the prediction. The one indicates the predicted category

predi = model1.predict(imagesTest[:])
predi

prediF = [[k+1, [0,0,0,0,0]] for k in range(len(predi))]

for i in range(len(predi)):
    prediF[i][1][np.argmax(predi[i])] = 1

prediF               


# In[14]:


from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# deletes the existing model


# In[22]:


# returns a compiled model
# identical to the previous one, to load a previous model and test it without recreating it
model1 = load_model('my_model.h5')


# In[16]:


#example of prediction with the new model
print(model1.predict(test_X[:1]))
print(test_Y[:1])

