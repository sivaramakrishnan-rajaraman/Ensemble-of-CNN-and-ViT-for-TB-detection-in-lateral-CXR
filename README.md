# Ensemble-of-CNN-and-ViT-for-TB-detection-in-lateral-CXR
An ensemble of convolutional neural network and vision transformer models to improve TB detection in lateral chest radiographs

## Codes:

### ensemble.py: 
The code is organized into three sections:
1. Pretraining the ImageNet-pretrained ViT and CNN models on a large-scale collection of CXRs to convert the weight layers specific to the CXR modality
2. Finetuning the modality-specific pretrained models on the TB and normal CXR data.
3. Ensemble evaluation: simple, weighted (SLSQP method), max voting, and model merging

### visualization.py:
The following code is used to visualize the class selective relevance maps from the CNN models and the attention maps from the ViT models and then construct an ensemble
of the attention and class selective relevance maps using the Sequential Least Squares Programming (SLSQP) algorithmic method that performs several iterations of constrained logarithmic loss minimization to converge to the optimal weights for the model interpretations.

## Requirements:

keras==2.6.0

matplotlib==3.5.0

numpy==1.19.5

opencv_python==4.5.4.58

pandas==1.3.4

Pillow==8.4.0

scikit_learn==1.0.1

scikit_plot==0.3.7

scipy==1.7.2

seaborn==0.11.2

tensorboard==2.6.0

tensorflow==2.6.2

tensorflow_addons==0.15.0

tensorflow_probability==0.15.0

tqdm==4.62.3

vit_keras==0.1.0
