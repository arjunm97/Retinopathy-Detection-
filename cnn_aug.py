# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:57:27 2020

@author: Arjun
"""

# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential, load_model,model_from_json
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense ,BatchNormalization
from keras import backend as K 
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import os
import pandas as pd 
from sklearn.datasets import make_circles
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
img_width, img_height = 224, 224

train_data_dir = 'F:/cnn_dr/dataset/train'
validation_data_dir ='F:/cnn_dr/dataset/testttttttttttttt'
val_dir='F:/8th sem/final_dataset_binary/Test'
nb_train_samples = 112
nb_validation_samples = 30
epochs = 100
batch_size = 10

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(2))
model.add(Activation(tf.nn.softmax))

model.compile(loss ='categorical_crossentropy', 
					optimizer ='rmsprop', 
				metrics =['accuracy']) 

train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
			horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='categorical') 

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='categorical') 

checkpoint=ModelCheckpoint("best_model.hdf5",monitor='loss',verbose=1,save_best_only=True,mode='auto',period=1)

model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size,callbacks=[checkpoint])

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode =None,shuffle = False) 
#model.load_weights("best_model.hdf5")
pre=model.predict_generator(validation_generator);
preds_cls_idx = pre.argmax(axis=-1);
pre_test_label=np.array(preds_cls_idx)
#label_test = np.load('/content/drive/My Drive/cnn_final_models/test.npy');
#label_test=[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3]#for fundas
label_test=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#label_test=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
precision = precision_score(label_test, preds_cls_idx,average=None)
print('Precision: %f' % np.max(precision))
recall = recall_score(label_test, preds_cls_idx,average=None)
print('Recall: %f' % np.max(recall))
f1 = f1_score(label_test, preds_cls_idx,average=None)
print('F1 score: %f' % np.max(f1))
matrix = confusion_matrix(label_test, preds_cls_idx)
print(matrix)
target_names=['class 0','class 1']
print(classification_report(label_test,preds_cls_idx,target_names=target_names))
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

y_true = label_test=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]




pre=model.predict_generator(validation_generator);
preds_cls_idx = pre.argmax(axis=-1);
pre_test_label=np.array(preds_cls_idx)
y_probas = pre_test_label

 
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=0)

# Print ROC curve
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
