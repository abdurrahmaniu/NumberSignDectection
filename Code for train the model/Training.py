




# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HpaSlL2RB72NnMsiaRip-1I1rf3PgasP
"""

#!rmdir data
#from google.colab import files
#files.upload()

#!unzip data.zip



from imageio import imread
img=imread('data/train/1/14.jpg')
print(img.shape)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten

model=Sequential()

model.add(Convolution2D(20,(3,3),input_shape=(250,200,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(30,(4,4),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(40,(4,4),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator(rescale=1/255,horizontal_flip=True,width_shift_range=0.2, height_shift_range=0.2)
val_gen=ImageDataGenerator(rescale=1/255)

train_set=train_gen.flow_from_directory('data/train',target_size=(250,200),batch_size=10,class_mode='categorical',color_mode='grayscale')
val_set=val_gen.flow_from_directory('data/test',target_size=(250,200),batch_size=10,class_mode='categorical',color_mode='grayscale')

from keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint('model2.h5',monitor='val_ccuracy',save_best_only=False,save_weights_only=False,save_freq='epoch')

model_info=model.fit(train_set,epochs=50,validation_data=val_set,steps_per_epoch=len(train_set),callbacks=[checkpoint])