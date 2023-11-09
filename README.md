# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Dataset
![281447376-326d553d-44e4-45f0-ad5c-f463b8aee99a](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/784f2387-9580-47e6-8f00-0b5f6d43bb95)

## Neural Network Model
![281447487-9a8d8e96-48f1-4715-8e40-7b58c96e24ee](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/1412be4d-b510-4e7c-8637-19f76f1a5fe6)


## DESIGN STEPS

### Step 1: Import Libraries

### Step 2: Allow GPU Processing

### Step 3: Read Images and Check Dimensions

### Step 4: Image Generator

### Step 5: Build and Compile the CNN Model

### Step 6: Train the Model

### Step 7: Plot the Training and Validation Loss

### Step 8: Evaluate the Model

### Step 9: Check for New Image

## PROGRAM
```
Program by : MOHAMED ATHIL B
Register.no : 212222230081
```
### Import Liraries:
```
import os
import random as rnd

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
```
### Allow GPU Processing:
```
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement 
sess = tf.compat.v1.Session(config=config)
set_session(sess)
```
### Read Images:
```
my_data_dir = "./cell_images"
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]
para_img= imread(train_path+'/parasitized/'+os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)
```
### Checking the image dimensions:
```
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
```
### Image Generator:
```
image_gen=ImageDataGenerator(rotation_range=20,width_shift_range=0.10,
				height_shift_range=0.10,rescale=1/255,shear_range=0.1,zoom_range=0.1,
				horizontal_flip=True, fill_mode='nearest')
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```
### DL Model - Build & Compile:
```
model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Dense(64,ativation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
batch_size = 16
```
### Fit the model:
```
train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],
                              color_mode='rgb',batch_size=batch_size,class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2],
                              color_mode='rgb',batch_size=batch_size,
                              class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=2,validation_data=test_image_gen)
model.save('cell_model.h5')
```
### Plot the graphs:
```
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
```
### Evaluate Metrics:
```
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```
### Check for New Image:
```
list_dir=["Un Infected","parasitized"]
dir_=(rnd.choice(list_dir))
p_img=imread(train_path+'/'+dir_+'/'+os.listdir(train_path+'/'+dir_)[rnd.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(p_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")
			+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![281447923-46516600-a49f-4db9-a7be-8dc61deec0d3](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/140b62eb-0255-4cc3-98f5-535674635f9a)

### Classification Report

![281448009-461cc427-09aa-4e1a-94dc-7c00a228487e](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/4c49489d-dfb7-4cf3-bb43-08f4ed8a85ee)

### Confusion Matrix
![281448054-ae04e288-bc0a-4e9d-a672-92a513fea124](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/9d214af0-b771-418b-b741-15018e686090)


### New Sample Data Prediction
![281448158-a4404cbb-0d54-4797-a362-2b9192707029](https://github.com/Bmohamedathil/malaria-cell-recognition/assets/119560261/f53b3c65-7063-4d78-be52-4576c8a50dc1)


## RESULT
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
