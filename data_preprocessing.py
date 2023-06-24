import pandas as pd

classes = {
  "pedestrian": 0,
  "rider": 7,
  "car": 1,
  "truck": 2,
  "bus": 3,
  "train": 4,
  "motorcycle": 5,
  "bicycle": 6,
  "traffic light": 8,
  "traffic sign": 9,
  "other vehicle": 10,
  "other person": 11,
  "trailer": 12,
}

# Read json labels
data_df=pd.read_json('bdd10k_labels_images.json')

import os
import shutil

image_width=1280
image_height=720

source_folder=""
destination_folder=""

# Convert json labels to yolo label format
for index,row in data_df.iterrows():
  
  labels_yolo=""

  text_file_name=row['name'].replace('jpg','txt')

  lista=row['labels']

  for id in lista:

    category=classes[id['category']]
    
    # Classes 7 and above had a significantly low representation, thus they were excluded from the dataset
    if(category>6):
      continue
    
    x1=id['box2d']['x1']
    x2=id['box2d']['x2']
    y1=id['box2d']['y1']
    y2=id['box2d']['y2']

    x_center_norm=((x1/image_width)+(x2/image_width))/2
    y_center_norm=((y1/image_height)+(y2/image_height))/2
    object_width_norm=(x2/image_width)-(x1/image_width)
    object_height_norm=(y2/image_height)-(y1/image_height)
    
    # If the object is too small discard it
    if(object_width_norm<0.035 or object_height_norm<0.035 ):
      continue

    labels_yolo+=str(category)+" "+str(x_center_norm)+" "+str(y_center_norm)+" "+str(object_width_norm)+" "+str(object_height_norm)+"\n"

    with open(text_file_name,"w") as file:
      file.write(labels_yolo)
 
  print(labels_yolo)