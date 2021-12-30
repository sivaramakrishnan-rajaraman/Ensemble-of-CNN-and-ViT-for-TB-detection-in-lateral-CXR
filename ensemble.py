'''
The following code is organized into three sections:
1. Pretraining the ImageNet-pretrained ViT and CNN models on a large-scale
collection of CXRs to convert the weight layers specific to the CXR modality
2. Finetuning the modality-specific pretrained models on the 
TB and normal CXR data.
3. Ensemble evaluation: simple, weighted (SLSQP method), max voting, model merging
'''
#%%
'''
Part 1: 
Pretraining the ImageNet-pretrained ViT and CNN models on a large-scale
collection of CXRs to convert the weight layers specific to the CXR modality
'''
#%%
# import libraries

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #if having multiple GPUs in the system 

#%%
#clear warnings and session

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) 
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

#%%
#import other libraries

import struct
from tensorflow.keras.preprocessing.image import img_to_array
import statistics
import keras_efficientnet_v2
import zlib
import time
import glob
import itertools
from itertools import cycle
from matplotlib import pyplot
import numpy as np
from numpy import sqrt, argmax, genfromtxt
from scipy import interp
from scipy.optimize import minimize
import pandas as pd
import math
import cv2
import statistics
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, Flatten, Concatenate, ZeroPadding2D, GlobalAveragePooling2D, Dense
from sklearn import metrics
from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow_addons as tfa

#%%
#custom function for class selective relevance mapping

def Generate_CRM_2Class(thisModel, thisImg_path, Threshold):             # generate Class Revlevance Map (CRM)
    try:
        # preprocess input image      
        width, height = thisModel.input.shape[1:3].as_list()
        original_img = cv2.imread(thisImg_path) 
        resized_original_image = cv2.resize(original_img, (width,height))        
    
        input_image = img_to_array(resized_original_image)
        input_image = np.array(input_image, dtype="float") /255.0       
        input_image = input_image.reshape((1,) + input_image.shape)
    
        class_weights = thisModel.layers[-1].get_weights()[0]
    
        get_output = K.function([thisModel.layers[0].input], [thisModel.layers[-3].output, #change this based on the deepest convolutional layer
                                 thisModel.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_image])
        conv_outputs = conv_outputs[ 0, :, :, :]     
        final_output = predictions   
        
        wf0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
        wf1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
    
        for i, w in enumerate(class_weights[:, 0]):     
            wf0 += w * conv_outputs[:, :, i]           
        S0 = np.sum(wf0)           # score at node 0 in the final output layer
        for i, w in enumerate(class_weights[:, 1]):     
            wf1 += w * conv_outputs[:, :, i]             
        S1 = np.sum(wf1)           # score at node 1 in the final output layer
    
        #Calculate incremental MSE
        iMSE0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        iMSE1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
        row, col = wf0.shape
        for x in range (row):
                for y in range (col):
                        tmp0 = np.array(wf0)
                        tmp0[x,y] = 0.                   # remove activation at the spatial location (x,y)
                        iMSE0[x,y] = (S0 - np.sum(tmp0)) ** 2
    
                        tmp1 = np.array(wf1)
                        tmp1[x,y] = 0.                  
                        iMSE1[x,y] = (S1 - np.sum(tmp1)) ** 2
         
      
        Final_crm = iMSE0 + iMSE1       # consider both positive and negative contribution
    
        Final_crm /= np.max(Final_crm)    # normalize
        resized_Final_crm = cv2.resize(Final_crm, (height, width)) # upscaling to original image size

        The_CRM = np.array(resized_Final_crm)
        The_CRM[np.where(resized_Final_crm < Threshold)] = 0.  # clean-up (remove data below threshold)

        return[resized_original_image, final_output, resized_Final_crm, The_CRM]
    except Exception as e:
        raise Exception('Error from Generate_CRM_2Class(): ' + str(e))

#%%
# custom function to find max mode

def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

#%%
# custom function to increase the DPI of the image

def writePNGwithdpi(im, filename, dpi=(72,72)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   # s = buffer.tostring()
   s = buffer.tobytes()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])
      
#%%
#custom functions to generate confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
# custom function to compute performance metrics

def matrix_metrix(real_values,pred_values,beta):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN**2 + FN * TN + FP**2 + FP * TN)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    Recall     = round( TP / (TP+FN),4 )
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    mat_met = pd.DataFrame({'Metric':['TP','TN','FP','FN','Accuracy','Precision','Recall','F1','MCC','Kappa'],
                            'Value':[TP,TN,FP,FN,Accuracy,Precision,Recall,F1,MCC,Kappa]})
    return (mat_met)

#%%
# load data: pretrain on a large-scale collection of CXR data from Padchest and Chexpert collections

img_width, img_height = 224,224
train_data_dir = "data/padchest_tb/train"
test_data_dir = "data/padchest_tb/test"
epochs = 64 
batch_size =16 # vary based on GPU capacity
num_classes = 2 # no-finding, abnormal
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
# declare image data generators

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) 

test_datagen = ImageDataGenerator(
        rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical', 
        subset = 'training')

validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical', 
        subset = 'validation')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical')

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

# ground truth labels
Y_test=test_generator.classes
print(Y_test.shape)
Y_test1=to_categorical(Y_test, 
                       num_classes=num_classes, 
                       dtype='float32')
print(Y_test1.shape)

#%%
#declare model architecture

# ViT models: - B/16 model

vit_model = vit.vit_b16(
        image_size = img_width,        
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        weights="imagenet21k+imagenet2012",
        classes = num_classes)

vit_model.summary()
out = tf.keras.layers.Flatten()(vit_model.output)
out = tf.keras.layers.Dense(num_classes, 'softmax')(out)
model_b16 = tf.keras.Model(inputs = vit_model.input, 
                        outputs = out,
                        name = 'vit_b16_retrain')
model_b16.summary()

#%%

# ViT - B/32 model

vit_model = vit.vit_b32(
        image_size = img_width,        
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        weights="imagenet21k+imagenet2012",
        classes = num_classes)

vit_model.summary()
out = tf.keras.layers.Flatten()(vit_model.output)
out = tf.keras.layers.Dense(num_classes, 'softmax')(out)
model_b32 = tf.keras.Model(inputs = vit_model.input, 
                        outputs = out,
                        name = 'vit_b32_retrain')
model_b32.summary()

#%%
# ViT - L/16 model

vit_model = vit.vit_l16(
        image_size = img_width,        
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        weights="imagenet21k+imagenet2012",
        classes = num_classes)

vit_model.summary()
out = tf.keras.layers.Flatten()(vit_model.output)
out = tf.keras.layers.Dense(num_classes, 'softmax')(out)
model_l16 = tf.keras.Model(inputs = vit_model.input, 
                        outputs = out,
                        name = 'vit_l16_retrain')
model_l16.summary()

#%%
# ViT - L/32 model

vit_model = vit.vit_l32(
        image_size = img_width,        
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        weights="imagenet21k+imagenet2012",
        classes = num_classes)

vit_model.summary()
out = tf.keras.layers.Flatten()(vit_model.output)
out = tf.keras.layers.Dense(num_classes, 'softmax')(out)
model_l32 = tf.keras.Model(inputs = vit_model.input, 
                        outputs = out,
                        name = 'vit_l32_retrain')
model_l32.summary()

#%%
#CNN models: VGG-16
  
model = VGG16(include_top=False, weights='imagenet', 
                        input_shape=input_shape)
model1 = Model(inputs=model.input, 
                      outputs=model.get_layer('block5_conv3').output)
x = model1.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
model_vgg16 = Model(inputs=model1.input, 
                    outputs=predictions, 
                    name='VGG16_pretrain_224')
model_vgg16.summary()

#%%
#CNN models: DenseNet-121 model
  
model = DenseNet121(input_shape=input_shape, include_top=False, weights ="imagenet")   

model1 = Model(inputs=model.input, 
                      outputs=model.get_layer('pool3_pool').output)
x = model1.output
x = ZeroPadding2D()(x)
x = Conv2D(256, (3,3), padding="valid", activation = 'relu')(x)   
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
model_d121 = Model(inputs=model1.input, 
                    outputs=predictions, 
                    name='D121_pretrain_224')
model_d121.summary()

#%%

#CNN models: EfficientNet-V2-B0 model
  
model1 = keras_efficientnet_v2.EfficientNetV2B0(input_shape=input_shape,
                                                pretrained="imagenet")  
model1.summary()
model2 = Model(inputs=model1.input, 
                      outputs=model1.get_layer('add_7').output)
x = model2.output
x = ZeroPadding2D()(x)
x = Conv2D(256, (3,3), padding="valid", activation = 'relu')(x)   
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
model_efv2b0 = Model(inputs=model2.input, 
                    outputs=predictions, 
                    name='EFV2B0_pretrain_224')
model_efv2b0.summary()

#%%
# here we show how to train a single model, repeat for other models
#enumerate and print layer names for VitT-B/16 model

for i, layer in enumerate(model_b16.layers):
    print(i, layer.name)

#%%
# compute class weights

train_classes = train_generator.classes
class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(train_classes),
                                     y = train_classes)
class_weights = dict(zip(np.unique(train_classes), class_weights)),
print(class_weights)

#%%

# declare optimizer and compile

sgd = SGD(learning_rate=0.0001, 
          momentum=0.9)  
model_b16.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 

#%%

# use calbacks and store model

filepath = 'weights/pretraining/' + model_b16.name +\
            '.{epoch:02d}-{val_accuracy:.4f}.h5' # path to the stored model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='min', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=10, 
                              verbose=1, 
                              mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='min', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#%%
#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
model_b16_history = model_b16.fit(train_generator, 
                          steps_per_epoch=nb_train_samples // batch_size,
                          epochs=epochs, 
                          validation_data=validation_generator,
                          callbacks=callbacks_list,
                          class_weight = class_weights,
                          validation_steps=nb_validation_samples // batch_size, 
                          verbose=1)

print('Training time: %s' % (time.time()-t))

#%%
# plot performance

N = epochs #change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_b16_history.history["loss"], 
         'orange', 
         label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_b16_history.history["val_loss"], 
         'red', 
         label="val_loss")
plt.plot(np.arange(1, N+1), 
          model_b16_history.history["accuracy"], 
          'blue', 
          label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_b16_history.history["val_accuracy"], 
         'green', 
         label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("vit_b16_pretrain_performance.png")

#%%

# load the trained model for inference, repeat for other models
model = load_model('weights/pretraining/x.h5', compile=False) # path to your stored model
model.summary()

#%%
# Predict on the test data

test_generator.reset() 
custom_y_pred = model.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
#we need the scores of only the positive abnormal class

custom_y_pred1 = custom_y_pred[:,1]

#%%
#print all metrics

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%

# print the confusion matrix 

target_names = ['No-finding', 'Abnormal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=5)

x_axis_labels = ['No-finding', 'Abnormal']  
y_axis_labels = ['No-finding', 'Abnormal'] 
plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
#plot the ROC curves 

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred[:,1])
auc_score=roc_auc_score(Y_test, custom_y_pred[:,1])
print(auc_score)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='ViT-B/16')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# Plot PR curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 custom_y_pred[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 custom_y_pred[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))

# plot the PR curve for the model
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], 
            linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', 
            color='red', 
            label='ViT-B/16')
# axis labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
'''
Step 2: Finetuning the CXR modality-specific pretrained models on the TB data.
Instantiate the pretrained models with their
modality-specific weights, add the classification layers,
and finetuned all the layers using a small learning rate on the 
TB/normal data. Here, we show for a single model, repeat the process
for other models
'''
#%%
# load data: TB versus no finding data

img_width, img_height = 224,224
train_data_dir = "data/train"
test_data_dir = "data/test"
epochs = 64 
batch_size = 16
num_classes = 2 # no-finding/TB
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
# declare image data generators

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) 

test_datagen = ImageDataGenerator(
        rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical', 
        subset = 'training')

validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical', 
        subset = 'validation')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical')

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

Y_test=test_generator.classes
print(Y_test.shape)

Y_test1=to_categorical(Y_test, 
                       num_classes=num_classes, 
                       dtype='float32')
print(Y_test1.shape)

#%%
# load the CXR modality-specific pretrained model, 
#here we show an example using the 
# ViT B16 model, repeat the process to finetune other models

model_b16.load_weights('weights/pretraining/vit_b16_retrain.h5') # path to your pretrained model
model_b16.summary()

base_model=Model(inputs=model_b16.input,
                        outputs=model_b16.get_layer('ExtractToken').output)
x = base_model.output  
x = tf.keras.layers.Flatten()(x) 
logits = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
model_b16f = Model(inputs=base_model.input, 
                    outputs=logits, 
                    name = 'vit_b16_finetune_224')
model_b16f.summary()

#%%
# compute class weight

train_classes = train_generator.classes
class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(train_classes),
                                     y = train_classes)
class_weights = dict(zip(np.unique(train_classes), class_weights)),
print(class_weights)

#%%
# declare optimizer

sgd = SGD(learning_rate=0.0001,  
          momentum=0.9)  
model_b16f.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 

#%%
filepath = 'weights/finetuning/' + model_b16f.name +\
            '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='min', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=10, 
                              verbose=1, 
                              mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='min', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#%%
#reset generators

train_generator.reset()
validation_generator.reset()

#train the model
modelb16f_history = model_b16f.fit(train_generator, 
                          steps_per_epoch=nb_train_samples // batch_size,
                          epochs=epochs, 
                          validation_data=validation_generator,
                          callbacks=callbacks_list,
                          class_weight = class_weights,
                          validation_steps=nb_validation_samples // batch_size, 
                          verbose=1)

print('Training time: %s' % (time.time()-t))

#%%
# plot performance

N = epochs #change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         modelb16f_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         modelb16f_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
          modelb16f_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         modelb16f_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("b16_finetune_performance.png")
   
#%%
'''
Evaluate performance as we did for the pretrained models before

'''
#%%
'''
Ensemble construction:
We construct the ensemble of the top-2, top-3, top-5 and top-7 models
using several ensemble techniques such as majority voting,
majority voting, simple averaging, weighted averaging, and model merging

'''
#%%
#top-1:
    
model1 = load_model('weights/finetuning/top_1.h5', #path to the top performing model and so on
                          compile=False)
model1.summary()

#measure performance on test data, 
test_generator.reset()
model1_y_pred = model1.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-2: 

model2 = load_model('weights/finetuning/top_2.h5', 
                          compile=False)
model2.summary()

#measure performance on test data, 
test_generator.reset()
model2_y_pred = model2.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-3: 

model3 = load_model('weights/finetuning/top_3.h5', 
                          compile=False)
model3.summary()

#measure performance on test data, 
test_generator.reset()
model3_y_pred = model3.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-4: 

model4 = load_model('weights/finetuning/top_4.h5', 
                          compile=False)
model4.summary()

#measure performance on test data, 
test_generator.reset()
model4_y_pred = model4.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-5: 

model5 = load_model('weights/finetuning/top_5.h5', 
                          compile=False)
model5.summary()

#measure performance on test data, 
test_generator.reset()
model5_y_pred = model5.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-6: 

model6 = load_model('weights/finetuning/top_6.h5', 
                          compile=False)
model6.summary()

#measure performance on test data, 
test_generator.reset()
model6_y_pred = model6.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-7: 

model7 = load_model('weights/finetuning/top_7.h5', 
                          compile=False)
model7.summary()

#measure performance on test data, 
test_generator.reset()
model7_y_pred = model7.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#lets do a dummy assignment of the predictions
model1_y_pred1 = model1_y_pred
model2_y_pred1 = model2_y_pred
model3_y_pred1 = model3_y_pred
model4_y_pred1 = model4_y_pred
model5_y_pred1 = model5_y_pred
model6_y_pred1 = model6_y_pred
model7_y_pred1 = model7_y_pred


#print the shape of the predictions
print("The shape of model1 prediction  = ", 
     model1_y_pred1.shape)
print("The shape of model2 prediction  = ", 
     model2_y_pred1.shape)
print("The shape of model3 prediction  = ", 
     model3_y_pred1.shape)
print("The shape of model4 prediction  = ", 
     model4_y_pred1.shape)
print("The shape of model5 prediction  = ", 
     model5_y_pred1.shape)
print("The shape of model6 prediction  = ", 
     model6_y_pred1.shape)
print("The shape of model7 prediction  = ", 
     model7_y_pred1.shape)

#%%
#compute argmax

model1_y_pred1 = model1_y_pred1.argmax(axis=-1)
model2_y_pred1 = model2_y_pred1.argmax(axis=-1)
model3_y_pred1 = model3_y_pred1.argmax(axis=-1)
model4_y_pred1 = model4_y_pred1.argmax(axis=-1)
model5_y_pred1 = model5_y_pred1.argmax(axis=-1)
model6_y_pred1 = model6_y_pred1.argmax(axis=-1)
model7_y_pred1 = model7_y_pred1.argmax(axis=-1)

#%%
# perform majority voting

#using top-2 models:
    
max_voting_2_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_2_pred = np.append(max_voting_2_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_2_pred1=to_categorical(max_voting_2_pred, 
                                  num_classes=num_classes, 
                                  dtype='float32')
print(max_voting_2_pred1.shape)

#%%
#using top-3 models:
    
max_voting_3_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_3_pred = np.append(max_voting_3_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i],
                                                 model3_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_3_pred1=to_categorical(max_voting_3_pred, 
                                  num_classes=num_classes, 
                                  dtype='float32')
print(max_voting_3_pred1.shape)

#%%
#using top-5 models:
max_voting_5_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_5_pred = np.append(max_voting_5_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i],
                                                 model3_y_pred1[i],
                                                 model4_y_pred1[i],
                                                 model5_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_5_pred1=to_categorical(max_voting_5_pred, 
                                  num_classes=num_classes, 
                                  dtype='float32')
print(max_voting_5_pred1.shape)

#%%
#using top-7 models:
max_voting_7_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_7_pred = np.append(max_voting_7_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i],
                                                 model3_y_pred1[i],
                                                 model4_y_pred1[i],
                                                 model5_y_pred1[i],
                                                 model6_y_pred1[i],
                                                 model7_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_7_pred1=to_categorical(max_voting_7_pred, 
                                  num_classes=num_classes, 
                                  dtype='float32')
print(max_voting_7_pred1.shape)

#%%

'''
evaluate the performance as done beforeduring modality-specific pretraining: 
Here we show how to measure performance using the majority voting of
the predictions of the top-2 models and report confusion matrix, 
AUC, and PR curves. Repeat for other max voting predictions.
'''
#%%

#we need the scores of only the positive abnormal class
max_voting_2_pred11 = max_voting_2_pred1[:,1]

#%%
#print all metrics

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      max_voting_2_pred1.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
# print the confusion matrix

target_names = ['No-finding', 'TB'] #vary the labels for another imaging modality
print(classification_report(Y_test1.argmax(axis=-1),
                            max_voting_2_pred1.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              max_voting_2_pred1.argmax(axis=-1))
np.set_printoptions(precision=5)
x_axis_labels = ['No-finding', 'TB']  
y_axis_labels = ['No-finding', 'TB'] 
plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
# measure Brier and Log losses

print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,max_voting_2_pred1[:,1]),4))

#compute Log loss

print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,max_voting_2_pred1[:,1]),4))

#%%
#plot the ROC curves 

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 max_voting_2_pred1[:,1])
auc_score=roc_auc_score(Y_test, max_voting_2_pred1[:,1])
print(auc_score)

fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='Top_2_max_voting')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# plot PR curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 max_voting_2_pred1[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 max_voting_2_pred1[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))

# plot the PR curve for the model
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=40)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', color='red', label='Top_2_max_voting')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
#save the predictions
predicted_class_indices=np.argmax(max_voting_2_pred1,axis=1)
print(predicted_class_indices)

'''
map the predicted labels with their unique ids such 
as filenames to find out what you predicted for which image.
'''

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#save the results to a CSV file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predicted_class_indices,
                      "Labels":predictions})
results.to_csv("maxvoting_top_2.csv",index=False)

#%%
# simple averaging
#top - 2 models:
    
average_pred_2=(model1_y_pred + model2_y_pred)/2
ensemble_model_2_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_2.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_2_averaging_accuracy)

#%%
#top - 3 models:
    
average_pred_3 = (model1_y_pred + model2_y_pred + model3_y_pred)/3
ensemble_model_3_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_3.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_3_averaging_accuracy)

#%%
#top-5 models:
    
average_pred_5=(model1_y_pred + model2_y_pred + \
                model3_y_pred + model4_y_pred + model5_y_pred)/5
ensemble_model_5_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_5.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_5_averaging_accuracy)

#%%
#top-7 models:
    
average_pred_7=(model1_y_pred + model2_y_pred + model3_y_pred + model4_y_pred +\
                model5_y_pred + model6_y_pred + model7_y_pred)/7
ensemble_model_7_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_7.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_7_averaging_accuracy)

#%%
# here we show measuring prediction performance
# using the averaging of the predictions of the top-2 models.
#repeat for other models. 

#we need the scores of only the positive abnormal class
average_pred_21 = average_pred_2[:,1]

#%%
#print all metrics

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      average_pred_2.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
# print the confusion matrix

target_names = ['No-finding', 'TB'] #vary the labels for another imaging modality
print(classification_report(Y_test1.argmax(axis=-1),
                            average_pred_2.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              average_pred_2.argmax(axis=-1))
np.set_printoptions(precision=5)
x_axis_labels = ['No-finding', 'TB']  
y_axis_labels = ['No-finding', 'TB'] 
plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
# measure Brier and log loss

print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,average_pred_21),4))

#compute Log loss

print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,average_pred_21),4))

#%%
#plot the ROC curves 

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 average_pred_2[:,1])
auc_score=roc_auc_score(Y_test, average_pred_2[:,1])
print(auc_score)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='Top_2_simple_averaging')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# plot PR curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 average_pred_2[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 average_pred_2[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))

# plot the PR curve for the model
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', color='red', label='Top_2_simple_averaging')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
#save the predictions

predicted_class_indices=np.argmax(average_pred_2,axis=1)
print(predicted_class_indices)

'''
map the predicted labels with their unique ids such 
as filenames to find out what you predicted for which image.
'''

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#save the results to a CSV file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predicted_class_indices,
                      "Labels":predictions})
results.to_csv("average_pred_top_2.csv",index=False)


#%%
#weighted averaging:
'''
Here, we calcualte the optimal weights for the models 
predictions through a constrained minimization process of the logarithmic loss 
function. We use the Sequential Least Squares Programming (SLSQP) 
algorithmic method that performs several iterations of 
constrained logarithmic loss minimization to converge 
to the optimal weights for the model predictions.
Here, we show for the top-2 models, repeat process for 
top-3, top-5, and top-7 models
'''
#%%
# create an arry of predictions
preds = [] # repeat for top-3, 5, and 7 models

test_generator.reset()
model1_y_pred = model1.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model1_y_pred)

test_generator.reset()
model2_y_pred = model2.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model2_y_pred)

# test_generator.reset()
# model3_y_pred = model3.predict(test_generator,
#                                     nb_test_samples // batch_size, 
#                                     verbose=1)
# preds.append(model3_y_pred)

# test_generator.reset()
# model4_y_pred = model4.predict(test_generator,
#                                     nb_test_samples // batch_size, 
#                                     verbose=1)
# preds.append(model4_y_pred)

# test_generator.reset()
# model5_y_pred = model5.predict(test_generator,
#                                     nb_test_samples // batch_size, 
#                                     verbose=1)
# preds.append(model5_y_pred)

# test_generator.reset()
# model6_y_pred = model6.predict(test_generator,
#                                     nb_test_samples // batch_size, 
#                                     verbose=1)
# preds.append(model6_y_pred)

# test_generator.reset()
# model7_y_pred = model7.predict(test_generator,
#                                     nb_test_samples // batch_size, 
#                                     verbose=1)
# preds.append(model7_y_pred)

#%%
#define a custom function to measure weighted accuracy

def calculate_weighted_accuracy(prediction_weights):
    weighted_predictions = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

#lets assume equal weights for the model predictions to begin with
# for the top-2 models
prediction_weights = [1. / 2] * 2 # change for top-3, 5, and 7 models
print(prediction_weights)
calculate_weighted_accuracy(prediction_weights)

#%%
# Create the loss metric 

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((nb_test_samples, num_classes), 
                                dtype='float32')
    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction
    return log_loss(Y_test1, final_prediction)

best_acc = 0.0
best_weights = None

# Parameters for optimization
constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

#%%
'''
now we determine how much weights we have to give
for each model prediction based on the log loss functions,
the process is repeated for 100 times to find the best combination
of weights for the ensemble models that results in
the highest accuracy and lowest loss
'''
#%%

NUM_TESTS = 100

for iteration in range(NUM_TESTS):
    
    prediction_weights = np.random.random(2) #change for top-3, 5, and 7 models
    
    # Minimise the loss 
    result = minimize(log_loss_func, 
                      prediction_weights, 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
    
    weights = result['x']
    weighted_predictions2 = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')  
    
    # Calculate weighted predictions
    for weight, prediction in zip(weights, preds):
        weighted_predictions2 += weight * prediction
    yPred = np.argmax(weighted_predictions2, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    
    # Calculate weight prediction accuracy
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)
    
    # Save current best weights 
    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights
        
    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)
calculate_weighted_accuracy(best_weights)

#%%
#use the predicted weights to compute the weighted predictions

prediction_weights = [0.66262907, 0.33737093] # weights measured from above
weighted_predictions2 = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')
for weight, prediction in zip(prediction_weights, preds):
    weighted_predictions2 += weight * prediction
yPred = np.argmax(weighted_predictions2, axis=1)
yTrue = Y_test1.argmax(axis=-1)
accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

#%%
# repeat the above process for the top-3, 5, and 7 models
# here we show how to measure performance using the weighted averaging
#predictions of the top-2 models

#%%
#we need the scores of only the positive abnormal class

weighted_predictions21 = weighted_predictions2[:,1]

#%%
#print all metrics

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      weighted_predictions2.argmax(axis=-1),
                      beta=0.4)
print (mat_met)


#%%
# print the confusion matrix

target_names = ['No-finding', 'TB'] #vary the labels for another imaging modality
print(classification_report(Y_test1.argmax(axis=-1),
                            weighted_predictions2.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              weighted_predictions2.argmax(axis=-1))
np.set_printoptions(precision=5)
x_axis_labels = ['No-finding', 'TB']  
y_axis_labels = ['No-finding', 'TB'] 
plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
#compute Brier and log loss

print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,weighted_predictions21),4))

#compute Log loss

print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,weighted_predictions21),4))

#%%
#plot the ROC curves 

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 weighted_predictions2[:,1])
auc_score=roc_auc_score(Y_test, weighted_predictions2[:,1])
print(auc_score)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='Top_2_weighted_averaging')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# plot PR curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 weighted_predictions2[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 weighted_predictions2[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))

no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', color='red', label='Top_2_weighted_averaging')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
'''
Next, we perform model-level ensembles. The top-2, 3, 5, and 7 models 
are selected. The ViT models are trunctated at the flatten layer.
The CNN models are truncated at the deepest convolutional layer
and added with a flatten layer.
These layers are then concatenated and appended with
a final dense layer to output prediction probabilities. 
Here, we show how we perform model-level ensembles using the top-2
performing models 
'''
#%%
#load each model

# top-performer:
    
model1 = load_model('weights/finetuning/top_1.h5', 
                    compile=False)
model1.summary() 
model1 = Model(inputs=model.input, 
                      outputs=model.get_layer('conv2d').output)
x1 = model1.output
x1 = Flatten()(x1)
model1v = Model(inputs=model1.input, outputs=x1)

#%%
# model-2:
    
model2 = load_model('weights/finetuning/top_2.h5', 
                          compile=False)
model2.summary()

#%%
model2=Model(inputs=model2.input,
                        outputs=model2.get_layer('flatten').output)
model2.summary()
x2 = model2.output
model2v = Model(inputs=model2.input, outputs=x2)

#%%

# model-3:
    
model3 = load_model('weights/finetuning/top_3.h5', 
                          compile=False)
model3.summary()

#%%
model3=Model(inputs=model3.input,
                        outputs=model3.get_layer('flatten').output)
model3.summary()
x3 = model3.output
model3v = Model(inputs=model3.input, outputs=x3)
    
#%%%%

# model-4:
    
model4 = load_model('weights/finetuning/top_4.h5', 
                          compile=False)
model4.summary()

#%%
model4 = Model(inputs=model4.input, 
                      outputs=model4.get_layer('conv2d').output)
x4 = model4.output
x4 = Flatten()(x4)
model4v = Model(inputs=model4.input, outputs=x4)

#%%

# model-5:
    
model5 = load_model('weights/finetuning/top_5.h5', 
                          compile=False)
model5.summary()

model5 = Model(inputs=model5.input, 
                      outputs=model5.get_layer('conv2d').output)
x5 = model5.output
model5v = Model(inputs=model5.input, outputs=x5)

#%%
# model-6:
    
model6 = load_model('weights/finetuning/top_6.h5', 
                          compile=False)
model6.summary()

#%%
model6 = Model(inputs=model6.input,
                        outputs=model6.get_layer('flatten').output)
model6.summary()
x6 = model6.output
model6v = Model(inputs=model6.input, outputs=x6)

#%%
# model-7:
model7 = load_model('weights/finetuning/top_7.h5', 
                          compile=False)
model7.summary()

#%%
model7 = Model(inputs=model7.input,
                        outputs=model7.get_layer('flatten').output)
model7.summary()
x7 = model7.output
model7v = Model(inputs=model7.input, outputs=x7)

#%%
#take the output of each model
  
out1 = model1v(model_input)    
out2 = model2v(model_input)  
out3 = model3v(model_input)
out4 = model4v(model_input)
out5 = model5v(model_input)
out6 = model6v(model_input)
out7 = model7v(model_input)

#%%
#concatenate the output of the top-2 models, repeat for other ensembles, top-3, 5, and 7 models
mergedOut = Concatenate()([out1,out2]) 
logits = Dense(num_classes, 
               activation='softmax', 
               name='predictions')(mergedOut)                   
model_merge2 = Model(inputs=model_input, 
                    outputs=logits, 
                    name = 'merge_top_2')
model_merge2.summary()

#%%

#print layer names and their numbers
{i: v for i, v in enumerate(model_merge2.layers)}

# print trainable layers
for l in model_merge2.layers:
    print(l.name, l.trainable)

#%%
#set trainable and non-trainable layers

# make everything until the final dense layer as non-trainable
for layer in model_merge2.layers[:5]: #change for the other model ensembles
    layer.trainable = False
for layer in model_merge2.layers[5:]:
    layer.trainable = True

# print trainable layers
for l in model_merge2.layers:
    print(l.name, l.trainable)    

#%%
#compile and train the merged model using a small initial learning rate

sgd = SGD(learning_rate=0.0001, momentum=0.9)  
model_merge2.compile(optimizer=sgd, 
                    loss='categorical_crossentropy',
                    metrics=['accuracy']) 

#%%
#begin training

filepath = 'weights/finetuning/' + \
    model_merge2.name + '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='max', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_accuracy', 
                              patience=10, 
                              verbose=1, 
                              mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='max', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#%%
#reset generators
train_generator.reset()
test_generator.reset()

#train the model
model_merge2_history = model_merge2.fit(train_generator, 
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs, 
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list, 
                                      validation_steps=nb_validation_samples // batch_size, 
                                      verbose=1)

print('Training time: %s' % (time.time()-t))

#%% plot performance

N = epochs # change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_merge2_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_merge2_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
         model_merge2_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_merge2_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("merge_top_2.png")

#%%
# load the trained model and evaluate performance, 
#keep compile as False since the model is used only for inference

model_merge2.load_weights('weights/finetuning/merge_top_2.h5')
model_merge2.summary()

#%%
#Generate predictions on the test data

test_generator.reset() 
custom_y_pred = model_merge2.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
#we need the scores of only the positive abnormal class

custom_y_pred1 = custom_y_pred[:,1]

#%%
#print all metrics

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#print brier and log loss scores

print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1),4))

print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,custom_y_pred1),4))

#%%
#plot the ROC curves 

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred[:,1])
auc_score=roc_auc_score(Y_test, custom_y_pred[:,1])
print(auc_score)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='top_2_model_merge')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# plot pr curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 custom_y_pred[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 custom_y_pred[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', color='red', label='top_2_model_merge')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
'''
END OF CODE

'''