import os
from skimage import io, util
import cv2
import numpy as np
# from PIL import Image,ImageEnhance
# import random

images_path = ''

images_dst_folder = ''
labels_dst_folder = ''

i=1
for image_name in os.listdir(images_path):
    
    image = cv2.imread(os.path.join(images_path,image_name))

    label_name = image_name.replace('.jpg','.txt')
    with open(label_name, 'r') as file:
        
        label = file.read()
    
    salt_image = util.random_noise(image, mode='salt', amount=0.05)
    salt_image = cv2.convertScaleAbs(salt_image, alpha=(255.0))
    cv2.imwrite(str(i)+'.jpg',salt_image)
    np.savetxt(str(i)+'.txt', label)
    i+=1
    
    blurred_image = cv2.GaussianBlur(image, (9, 9), 0)
    cv2.imwrite(str(i)+'.jpg',blurred_image)
    np.savetxt(str(i)+'.txt', label)
    i+=1

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(i)+'.jpg',gray_image)
    np.savetxt(str(i)+'.txt', label)
    i+=1

    



    

    