import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import glob
from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D
from tifffile import imsave
import matplotlib.pyplot as plt
import nibabel as nib
import splitfolders

scaler = MinMaxScaler()

# TRAIN_DATASET_PATH = 'BraTS2020_TrainingData'

# test_image_flair = nib.load('C:/Users/rohan/OneDrive/Desktop/OneDrive/Desktop/BraTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata()
# test_image_t1ce = nib.load('C:/Users/rohan/OneDrive/Desktop/OneDrive/Desktop/BraTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_t1ce.nii').get_fdata()
# test_image_t2 = nib.load('C:/Users/rohan/OneDrive/Desktop/OneDrive/Desktop/BraTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_t2.nii').get_fdata()
# print(test_image_flair.max())

# test_mask = nib.load('C:/Users/rohan/OneDrive/Desktop/OneDrive/Desktop/BraTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_seg.nii').get_fdata()
# test_mask=test_mask.astype(np.uint8)
# ## Exploring the data to combine the t2,t1ce,flair images to make a 3D Channeled Image
# combined_x = np.stack([test_image_flair,test_image_t1ce,test_image_t2],axis=3)

# #Cropping to a size to be divisible bt 64 so we can later extract 64x64x64 patches for training

# combined_x  = combined_x[56:184,56:184,13:141]
# test_mask = test_mask[56:184,56:184,13:141]

# n_slice = random.randint(0,test_mask.shape[2])
# plt.figure(figsize=(12,8))

# plt.subplot(221)
# plt.imshow(combined_x[:,:,n_slice,0],cmap='gray')
# plt.title("Image flair")
# plt.subplot(222)
# plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
# plt.title('Image t1ce')
# plt.subplot(223)
# plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
# plt.title('Image t2')
# plt.subplot(224)
# plt.imshow(test_mask[:,:,n_slice])
# plt.title('Mask')
# plt.show()

# np.save('BraTS/combined',combined_x)

#Let's apply the same preprocessing to all the images
t2_list_files = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list_files = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list_files = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list_files = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))


for img in range(len(t2_list_files)):
    print(f"Preparing Image and Mask number {img}")
    temp_img_t2 = nib.load(t2_list_files[img]).get_fdata()
    temp_img_t2 = scaler.fit_transform(temp_img_t2.reshape(-1,temp_img_t2.shape[-1])).reshape(temp_img_t2.shape)
    
    temp_img_t1ce = nib.load(t1ce_list_files[img]).get_fdata()
    temp_img_t1ce = scaler.fit_transform(temp_img_t1ce.reshape(-1,temp_img_t1ce.shape[-1])).reshape(temp_img_t1ce.shape)

    temp_img_flair = nib.load(flair_list_files[img]).get_fdata()
    temp_img_flair = scaler.fit_transform(temp_img_flair.reshape(-1,temp_img_flair.shape[-1])).reshape(temp_img_flair.shape)
 
    temp_mask = nib.load(mask_list_files[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3 # Reassign mask values 4 to 3
    
    temp_combined_images = np.stack([temp_img_flair,temp_img_t1ce,temp_img_t2],axis=3)
    
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val,counts = np.unique(temp_mask,return_counts=True)
    
    if (1-(counts[0]/counts.sum())) > 0.01: #At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask,num_classes=4)
        np.save('BraTS2020_TrainingData/input_data_3channels/images/image_'+str(img)+'.npy', temp_combined_images)
        np.save('BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(img)+'.npy', temp_mask)
    else:
        print("I am Useless")
        
        
input_folder = 'BraTS2020_TrainingData/input_data_3channels/'
output_folder = 'BraTS2020_TrainingData/input_data_128/'

splitfolders.ratio(input_folder,output_folder,seed=42,ratio=(.75,.25),group_prefix=None)
    
