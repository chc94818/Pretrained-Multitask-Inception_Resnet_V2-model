#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:27:44 2018

@author: shirhe-lyh
"""

"""Train a CNN classification model via pretrained ResNet-50 model.

Example Usage:
---------------
python3 train.py \
    --resnet50_model_path: Path to pretrained ResNet-50 model.
    --record_path: Path to training tfrecord file.
    --logdir: Path to log directory.
"""

import tensorflow as tf
import Data_Loader as ld
import Label_Encoder as le
import Data_Augmentation as dt_aug
import model
from os import walk
from os.path import join
from skimage import io
from matplotlib import pyplot as plt
import cv2,csv
DATA_NAME = "data_o_d_re"
DATA_PATH = "../Data/"+DATA_NAME
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
# 二進位資料
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 整數資料
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 浮點數資料
def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_data_list (path):
#train type :
    image_path = path+"/image"
    label_path = path+"/label"
    """Load data from `path`"""
    
    # label_gender  : 0->Male
    #                 1->Female 
    #
    # label age     : 0->Child
    #                 1->Teen
    #                 2->Adult
    #                 3->Middle
    #                 4->Senior
    name_list = []
    label_gender_list = []
    label_age_list = []
    for root, dirs, files in walk(image_path):
        for f in files:

            fullpath_image = root+'/'+ f
            fullpath_label = (root[0:-5]+'label/'+f[0:-4]+'.txt')
            
            f=open(fullpath_label,'r')
            lines=f.readline()
            
            gender = lines.split()[2]
            #print(gender)
            if gender == 'GenderMale':
                temp_gender = 0
            elif gender == 'GenderFemale':
                temp_gender = 1

            lines=f.readline()
            
            age = lines.split()[2]
            #print(age)
            if age == 'AgeChild':
                temp_age = 0
            #elif age == 'AgeTeen':
            #    label_age[file_count] = 1
            elif age == 'AgeAdult':
                temp_age = 1
            #elif age == 'AgeMiddle':
            #    label_age[file_count] = 3
            elif age == 'AgeSenior':
                temp_age = 2
            

            name_list.append(fullpath_image)
            label_gender_list.append(temp_gender)
            label_age_list.append(temp_age)
    return name_list, label_gender_list, label_age_list

def main(_):
    #dataset = get_record_dataset(FLAGS.record_path, num_samples=79573, num_classes=54)
                                 
    #data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    #image, label = data_provider.get(['image', 'label'])
    
    # Data augumentation
    #image = tf.image.random_flip_left_right(image)                                                             
    """
    image_ = tf.placeholder(tf.float32, [None, n_input],name='image-input')
    image = tf.reshape(image_, shape=[-1, 224, 224, 3])

    label_gender_encode = tf.placeholder(tf.int32, [None, 2], name='label_gender-input')

    inputs, labels = tf.train.batch([image, label_gender_encode],
                                    batch_size=8,                                                                                                                                                                                           
                                    allow_smaller_final_batch=True)
    """



    # read image name list, and label list
    name_list, label_gender_list, label_age_list = get_data_list(DATA_PATH)
    #names = name_list[0:4]
    #labels = label_gender_list[0:4]

    # encode label
    #labels_encode = le.encode_labels(labels, 2)
    # data augmentation
    #images_aug = dt_aug.data_augmentation(datas,0)
    

    # TFRecords 檔案名稱
    tfrecords_train_filename = 'train.tfrecords'
    tfrecords_test_filename = 'test.tfrecords'
    # 建立 TFRecordWriter
    train_writer = tf.python_io.TFRecordWriter(tfrecords_train_filename)
    test_writer = tf.python_io.TFRecordWriter(tfrecords_test_filename)
    #print(len(names))
    #print(len(labels))
    file_counter = 0
    test_counter = 0
    train_counter = 0
    for path, label_gender, label_age in zip(name_list, label_gender_list, label_age_list):
        #print(path, label)
        file_counter+=1
        print("file processing : ", file_counter)
        image_temp = cv2.imread(path)  
        image_resize = cv2.resize(image_temp, (IMAGE_HEIGHT, IMAGE_WIDTH))  
        b,g,r = cv2.split(image_resize)  
        rgb_image = cv2.merge([r,g,b])
        #plt.imshow(rgb_image)
        #plt.show()
        # 取得圖檔尺寸資訊
        height, width, depth = rgb_image.shape
        #print("image : ", rgb_image.shape)
        #print("height : ", height)
        #print("width : ", width)
        #print("depth : ", depth)
        #plt.imshow(image)
        #plt.show()
        # 序列化資料
        image_string = rgb_image.tostring()
        # 建立包含多個 Features 的 Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_string': _bytes_feature(image_string),
            'label_gender': _int64_feature(label_gender),
            'label_age': _int64_feature(label_age)}))

        
        if(file_counter<=1000):
            test_counter += 1
            test_writer.write(example.SerializeToString())
        else:
            train_counter += 1
            train_writer.write(example.SerializeToString())

        
    # 關閉 TFRecordWriter
    test_writer.close()
    train_writer.close()
    print("total file : ", file_counter)
    print("test file : ", test_counter)
    print("train file : ", train_counter)
if __name__ == '__main__':
    tf.app.run()
    print("height : ", height)
