## We face a problem here
## Keras can't really understand .npy files. It understands .jpg,.png,.tff files for images.

import os
import numpy as np
import matplotlib.pyplot as plt
import random

def load_image(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            image = np.load(img_dir+image_name)          
            images.append(image)
    images = np.array(images)
    return(images)

def image_loader(img_directory,img_list,mask_directory,mask_list,batch_size):
    L = len(img_list)
    while True:
        start = 0
        end = batch_size
        while start < L:
            print(f"Starting from {start}")
            limit = min(end,L)
            X = load_image(img_directory,img_list[start:limit])
            Y = load_image(mask_directory,mask_list[start:limit])
            
            yield(X,Y) #tuple of images and masks with batch_size samples
            
            start += batch_size
            end += batch_size
            
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
train_image_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 32

train_image_datagen = image_loader(train_img_dir,train_image_list,train_mask_dir,train_mask_list,batch_size)
img,msk = train_image_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask,axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()
