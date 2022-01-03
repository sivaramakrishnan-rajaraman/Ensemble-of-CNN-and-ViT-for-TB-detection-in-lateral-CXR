'''
The following code is used to visualize the
class selective relevance maps from the CNN models and the 
attention maps from the ViT models and then construct an ensemble
of the attention and class selective relevance maps
using the Sequential Least Squares Programming (SLSQP) 
algorithmic method that performs several iterations of 
constrained logarithmic loss minimization to converge 
to the optimal weights for the model interpretations.
'''

#%%
# import libraries
#clear warnings and session

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #if having multiple GPUs in the system 
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
# read source data that are to be interpreted 

source = glob.glob("data/test/tb/*.png")
source.sort()

image_size = 224

#%%
# load models:
# top performer:
    
model_d121 = load_model('weights/finetuning/D121_finetune_224.54-0.8750.h5', 
                          compile=False)
model_d121.summary()

#%%
# second performer:

model_b = load_model('weights/finetuning/vit_b32_finetune_224.32-0.8542.h5', 
                          compile=False)
model_b.summary()

model_b32=Model(inputs=model_b.input,
                        outputs=model_b.get_layer('ExtractToken').output)
model_b32.summary()

#%%
#  third performer:
    
model_l = load_model('weights/finetuning/vit_l16_finetune_224.09-0.8365.h5', 
                          compile=False)
model_l.summary()

model_l16=Model(inputs=model_l.input,
                        outputs=model_l.get_layer('ExtractToken').output)
model_l16.summary()

#%%
#  fourth performer:
    
model_vgg16 = load_model('weights/finetuning/VGG16_finetune_224.26-0.8542.h5', 
                          compile=False)
model_vgg16.summary()

#%%
# fifth performer:
    
model_eff = load_model('weights/finetuning/EF2B0_finetune_224.24-0.8229.h5', 
                          compile=False)
model_eff.summary() 

#%%
# sixth performer:

model_ll = load_model('weights/finetuning/vit_l32_finetune_224.11-0.7788.h5', 
                          compile=False)
model_ll.summary()

model_l32=Model(inputs=model_l.input,
                        outputs=model_l.get_layer('ExtractToken').output)
model_l32.summary()

#%%
# 7th performer:

model_bb = load_model('weights/finetuning/vit_b16_finetune_224.23-0.7812.h5', 
                          compile=False)
model_bb.summary()

model_b16=Model(inputs=model_l.input,
                        outputs=model_l.get_layer('ExtractToken').output)
model_b16.summary()   

#%%
# running the loop for individual models
# and generating class selective relevance maps for the
# CNN models

# first for the densenet-121 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_d121 = model_d121.predict(x)[0]    
    
    #print predictions
    print(preds_d121)          
   
    # compute attention maps    
    InImage1, OutScores1, aCRM_Img1, tCRM_Img1 = Generate_CRM_2Class(model_d121,
                                                                 f, 0.2) 
    aHeatmap_d121 = cv2.applyColorMap(np.uint8(255*aCRM_Img1), cv2.COLORMAP_JET)
    aHeatmap_d121[np.where(aCRM_Img1 < 0.2)] = 0 # threshold to remove noise
    
    #compute superimposed image
    superimposed_img_avg = aHeatmap_d121 * 0.4 + img #0.4 here is a heatmap intensity factor.
    
    writePNGwithdpi(superimposed_img_avg, 
                    "activations/d121/{}.png".format(img_name[:-4]), (300,300))    

#%%
# VGG16 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_vgg16 = model_vgg16.predict(x)[0]    
    
    #print predictions
    print(preds_vgg16)          
   
    # compute attention maps    
    InImage1, OutScores1, aCRM_Img1_vgg16, tCRM_Img1 = Generate_CRM_2Class(model_vgg16,
                                                                 f, 0.2) 
    aHeatmap_vgg16 = cv2.applyColorMap(np.uint8(255*aCRM_Img1_vgg16), cv2.COLORMAP_JET)
    aHeatmap_vgg16[np.where(aCRM_Img1_vgg16 < 0.2)] = 0
    
    #compute superimposed image
    superimposed_img_avg_vgg16 = aHeatmap_vgg16 * 0.4 + img 
    
    writePNGwithdpi(superimposed_img_avg_vgg16, 
                    "activations/vgg16/{}.png".format(img_name[:-4]), (300,300))   

#%%

# EfficientNet-V2-B0 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_eff = model_eff.predict(x)[0]    
    
    #print predictions
    print(preds_eff)          
   
    # compute attention maps    
    InImage1, OutScores1, aCRM_Img1_eff, tCRM_Img1 = Generate_CRM_2Class(model_eff,
                                                                 f, 0.2) 
    aHeatmap_eff = cv2.applyColorMap(np.uint8(255*aCRM_Img1_eff), cv2.COLORMAP_JET)
    aHeatmap_eff[np.where(aCRM_Img1_eff < 0.2)] = 0
    
    #compute superimposed image
    superimposed_img_avg_eff = aHeatmap_eff * 0.4 + img 
    
    writePNGwithdpi(superimposed_img_avg_eff, 
                    "activations/ef2b0/{}.png".format(img_name[:-4]), (300,300))   

#%%
# save the attention maps embedded on the original image
# using the ViT-B16 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_b16 = model_bb.predict(x)[0]
    
    #print predictions
    print(preds_b16) 
    
    #preprocess the image
    image1 = utils.read(img, image_size)
    
    # compute attention maps    
    # for Vit_B32 model
    attention_map_b16 = visualize.attention_map(model=model_b16, image=image1)
    aHeatmap_b16 = cv2.applyColorMap(np.uint8(255*attention_map_b16), cv2.COLORMAP_HOT)
    
    #compute superimposed image
    superimposed_img_avg_b16 = aHeatmap_b16 * 0.4 + img 
    
    writePNGwithdpi(superimposed_img_avg_b16, 
                    "activations/b16/{}.png".format(img_name[:-4]), (300,300)) 


#%%
# save the attention maps embedded on the original image
# using the ViT-B32 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_b32 = model_b.predict(x)[0]
    
    #print predictions
    print(preds_b32) 
    
    #preprocess the image
    image1 = utils.read(img, image_size)
    
    # compute attention maps    
    attention_map_b32 = visualize.attention_map(model=model_b32, image=image1)
    aHeatmap_b32 = cv2.applyColorMap(np.uint8(255*attention_map_b32), cv2.COLORMAP_HOT)

    #compute superimposed image
    superimposed_img_avg = aHeatmap_b32  * 0.4 + img 
    
    writePNGwithdpi(superimposed_img_avg, 
                    "activations/b32/{}.png".format(img_name[:-4]), (300,300)) 

#%%
# save the attention maps embedded on the original image
# using the ViT-L16 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_l16 = model_l.predict(x)[0]
    
    #print predictions   
    print(preds_l16) 
   
    #preprocess the image
    image1 = utils.read(img, image_size)
    
    # compute attention maps    
    attention_map_l16 = visualize.attention_map(model=model_l16, image=image1)
    aHeatmap_l16 = cv2.applyColorMap(np.uint8(255*attention_map_l16), cv2.COLORMAP_JET)
       
    #compute superimposed image
    superimposed_img_avg = aHeatmap_l16 * 0.4 + img 
    
    writePNGwithdpi(superimposed_img_avg, 
                    "activations/l16/{}.png".format(img_name[:-4]), (300,300))     

#%%
# save the attention maps embedded on the original image
# using the ViT-LB32 model

for f in source:
    img = load_img(f)
    img_name = f.split(os.sep)[-1] 
    
    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_l32 = model_ll.predict(x)[0]
    
    #print predictions
    print(preds_l32) 
    
    #preprocess the image
    image1 = utils.read(img, image_size)
    
    # compute attention maps     
    attention_map_l32 = visualize.attention_map(model=model_l32, image=image1)
    aHeatmap_l32 = cv2.applyColorMap(np.uint8(255*attention_map_l32), cv2.COLORMAP_HOT)
   
    #compute superimposed image
    superimposed_img_avg_l32 = aHeatmap_l32 * 0.4 + img 
    writePNGwithdpi(superimposed_img_avg_l32, 
                    "activations/l32/{}.png".format(img_name[:-4]), (300,300)) 
   
#%%
# running the loop and recording the average activations using the 
#SLSQP weights computed for the top-3 performing models

for f in source:
    img = load_img(f)
    #preprocess the image
    image1 = utils.read(img, image_size)
    img_name = f.split(os.sep)[-1] 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 
    
    #predict on the image
    preds_d121 = model_d121.predict(x)[0] # top performer
    print(preds_d121)
    preds_b32 = model_b.predict(x)[0] # 2nd top
    print(preds_b32) 
    preds_l16 = model_l.predict(x)[0] # 3rd top
    print(preds_l16)    
    
    # compute attention maps    
    # for DenseNet-121 model
    InImage1, OutScores1, attention_map_d121, tCRM_Img1 = Generate_CRM_2Class(model_d121,
                                                                 f, 0.2) 
        
    # for Vit_L16 model
    attention_map_l16 = visualize.attention_map(model=model_l16, image=image1)
    attention_map_l16 = cv2.resize(attention_map_l16, (image_size, image_size))
    
    # for Vit_B32 model
    attention_map_b32 = visualize.attention_map(model=model_b32, image=image1)
    attention_map_b32 = cv2.resize(attention_map_b32, (image_size, image_size))
    
    # heatmap_avg = (0.6631167 * attention_map_d121 + \
    #                0.14855253 * attention_map_l16 + \
    #                    0.1883307 * attention_map_b32)/3 #from SLQLP weights
    heatmap_avg = (0.67114851 * attention_map_d121 + 0.32885149 * attention_map_b32)/2 #from SLQLP weights    
    aHeatmap_avg = cv2.applyColorMap(np.uint8(255*heatmap_avg), cv2.COLORMAP_JET)
    aHeatmap_avg[np.where(heatmap_avg < 0.25)] = 0 # thresholded to remove noise
    
    #compute superimposed image
    superimposed_img_avg = aHeatmap_avg * 0.4 + img #0.4 here is a heatmap intensity factor.
    
    writePNGwithdpi(superimposed_img_avg, 
                    "activations/combined/{}.png".format(img_name[:-4]), (300,300))  
    
#%%
'''
Once we store these images with the activation masks embedded, we run the following matlab code
to perform the following:
    
1. Compute the difference image of the orginal and the image with the embedded activation mask.
This gives only the region of the activation.

2. Convert the activation region to a bounding box. 

3. Store the bounding box coordinates to a CSV file for evaluation.
'''
#%%
'''
clear
sheet={}; 
cam_folder_path = ['path\to\images with activation'];
img_folder_path = ['path\to\original_image'];
save_folder_path = ['path\to\saving\masks']; # this main folder contains two subfolders, masks_actual, and masks_bounding_boxes
save_folder_path_mask = ['path\to\saving\masks\masks_actual'];
save_folder_path_maskBB = ['path\to\saving\masks\masks bounding boxes'];

camfiles = dir(cam_folder_path);
imgfiles = dir(img_folder_path);
count=1;
for npat = 3:length(camfiles)
    patname = camfiles(npat).name;
    cam = imread([cam_folder_path filesep camfiles(npat).name]);
    img = imread([img_folder_path filesep imgfiles(npat).name]);

    diff_img = cam - img;
    gray_img = rgb2gray(diff_img);
    mask_img = gray_img>0;
    regs = regionprops(mask_img,'boundingbox');
    BB=[]; xi=[]; yi=[];
    for i=1:length(regs)
        BB(i,:) = regs(i).BoundingBox;   
        points = bbox2points(BB(i,:));
        xi = [xi points(:,1)];
        yi = [yi points(:,2)];
        sheet{count,1} = patname;
        for temp=2:5
            sheet{count,temp} = BB(i,temp-1);
        end
        count=count+1;
    end

    maskBB=[];
    for i=1:size(xi,2)
        maskBB(:,:,i) = poly2mask(xi(:,i),yi(:,i),size(mask_img,1),size(mask_img,2));
    end
    summaskBB = sum(maskBB,3);
    summaskBB(summaskBB>1) = 1;
    imwrite(mask_img,[save_folder_path_mask filesep patname],'png');
    imwrite(summaskBB,[save_folder_path_maskBB filesep patname],'png');

end

% Save output as CSV file
Final = cell2table(sheet,'VariableNames',{'Patient Name', 'xmin', 'ymin', 'width', 'height'});
output_csv_name = 'bbmask.csv';
writetable(Final, [save_folder_path filesep output_csv_name]);

'''
#%%
'''
Oce the above CSV file is computed and we also have the expert radiologist
annotations, we can compute the Kappa and average precision values at
different IoU thresholds (from 0.5 to 0.95 st increments of 0.05) as suggested in
the MS COCO challenge guidelines. 

'''
#%%
# custom function to get the mask from annotations 

def get_mask(img_name, df, h, w):
    im_csv_np = df.loc[:,"id"].values # this header contains the filenems of images {filename}.png
    idx = np.where(im_csv_np == img_name)
    if idx[0].shape[0]: # if there is a match shape[0] should 1, if not 0
        mask = np.zeros((len(idx[0]),h,w))
        for k,j in enumerate(idx[0]):
            i = j.item()
            # get location of x, y, width , and height
            mask[k,int(df.loc[i]['y']):int(df.loc[i]['y'])+int(df.loc[i]['height']),
                        int(df.loc[i]['x']):int(df.loc[i]['x'])+int(df.loc[i]['width'])] = 1.0
    else:
        mask = np.zeros((1,h,w))
    return mask

#%%

# custom function to measure other metrics

def jaccard_index(a,b): #same as IoU
    intersection = np.sum(np.multiply(a,b))
    union = np.sum(a+b) - intersection
    return intersection/union

def dice_score(a,b): #same as F-score 
    num = 2 * np.sum(np.multiply(a,b))
    den = np.sum(a) + np.sum(b)
    return num/den

#%%
# custom function to compute Kappa agreement based on IoU or Dice

def kappa_eval(ref,mask,thr, metric = 'jaccard'): #can also use dice score
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    if ref.any() or mask.any():
        for i in range(len(mask)):
            for j in range(len(ref)):
                if metric == 'jaccard':
                    iou_val = jaccard_index(mask[i],ref[j])
                else:
                    iou_val = dice_score(mask[i],ref[j])
                if iou_val >= thr:
                    TP += 1
                elif iou_val < thr and iou_val > 0:
                    FP += 1
        for k in range(len(mask)):
            if np.sum(np.multiply(mask[k],np.sum(ref,0))) == 0:
                FP += 1
        if len(ref) > TP+FP:
            FN += len(ref) - (TP+FP)
    else:
        TN = 1
    
    print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP,FP,FN,TN))
    print("*"*20)
    return TP,FP,FN,TN

#%%

# compute Kappa and other metrics

def eval_metric(TP, FP, FN, TN):
    po = (TP+TN)/(TP+FP+FN+TN)
    p_true = ((TP+FN)*(TP+FP))/((TP+FP+FN+TN)**2)
    p_false = ((FP+TN)*(FN+TN))/((TP+FP+FN+TN)**2)
    pe = p_true + p_false
    # Cohen's Kappa
    if pe == 1:
        k = 1
    else:
        k = (po - pe)/(1-pe)
    return k

def sensitivity_metric(TP, FP, FN, TN):
    pnum = TP/(TP + FN)
    return pnum

def precision_metric(TP, FP, FN, TN):
    pnum = TP/(TP + FP)
    return pnum

def specificity_metric(TP, FP, FN, TN):
    pnum = TN/(TN + FP)
    return pnum

def ppv_metric(TP, FP, FN, TN):
    pnum = TP/(TP + FP)
    return pnum

def npv_metric(TP, FP, FN, TN):
    pnum = TN/(TN + FN)
    return pnum

#%%
# evaluate metrics

metric = 'jaccard' # ['jaccard','dice']
iou_thr = 0.9 # [0.5 t0 0.95 in increments of 0.05]

#path to images
filenames = glob.glob("path/to/images/*.png") #path to images
filenames.sort()

#ground truth annotation
df1 = pd.read_csv('gt_mask.csv')

#from the model ensemble averaging masks
df2 = pd.read_csv('pred_mask.csv')

#declare variables
TP = 0
FP = 0
FN = 0
TN = 0

for f in filenames:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    h,w = img.shape
    img_name = f.split(os.sep)[-1]
    print(img_name)
    mask1 = get_mask(img_name, df1, h, w) # ground truth annotation
    mask2 = get_mask(img_name, df2, h, w) # student annotation
    TP_,FP_,FN_, TN_ = kappa_eval(mask1, mask2, iou_thr, metric) #mask1 is the ground truth     
    TP += TP_
    FP += FP_
    FN += FN_
    TN += TN_
        
print("Total")
print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP,FP,FN,TN))
print("*"*20)
kappa = eval_metric(TP, FP, FN, TN)
print("kappa: ",kappa)
sensitivity = sensitivity_metric(TP, FP, FN, TN)
print("Sensitivity: ",sensitivity)
precision = precision_metric(TP, FP, FN, TN)
print("Precision: ",precision)
specificity = specificity_metric(TP, FP, FN, TN)
print("Specificity: ",specificity)
PPV = ppv_metric(TP, FP, FN, TN)
print("Positive predictive value: ",PPV)
NPV = npv_metric(TP, FP, FN, TN)
print("Negative predictive value: ",NPV) 

#%%
'''
END OF CODE

'''
