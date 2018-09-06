import tensorflow as tf
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2
import sys
import inception_resnet_v2_multimask as infer
import inception_preprocessing
import Data_Loader_nctu as ld
import Label_Encoder as le
import Data_Augmentation as dt_aug
import MinMax_Scaler as mms
import time
import gc 
import skimage.io as io
# Parameters

# TFRecords 檔案名稱
tfrecords_filename = 'train.tfrecords'

DATA_NAME = "tfr"
DATA_PATH = "../Data/"+DATA_NAME
MODEL_SAVE_PATH = './pretrain_test/'
MODEL_NAME = "inception_resnet_v2"
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.995

checkpoints_dir = '/Multi-Task_CNN/NN/checkpoints/'
checkpoints_filename = 'inception_resnet_v2.ckpt'
model_name = 'InceptionResnetV2'
save_step = 20

# Train setting
BATCH_SIZE = 16
MIN_KM = 5
EPOCH = 100
display_step = 1
save_step = 20
DECAY_STEPS = 100*BATCH_SIZE
FIRST_TRAIN = True

# Network Parameters
GENDER_CLASS_NUM = 2
AGE_CLASS_NUM = 3
n_input = 89401


# image setting
CROP_HEIGHT= 299
CROP_WIDTH = 299
NUM_CHANNELS = 3

slim = tf.contrib.slim
def get_data_size():
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    counter = 0
    for string_record in record_iterator:
       counter += 1
    return counter

def read_and_decode(filename_queue):
    # 建立 TFRecordReader
    reader = tf.TFRecordReader()

    # 讀取 TFRecords 的資料
    _, serialized_example = reader.read(filename_queue)

    # 讀取一筆 Example
    features = tf.parse_single_example(
        serialized_example,
        features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_string': tf.FixedLenFeature([], tf.string),
        'label_gender': tf.FixedLenFeature([], tf.int64),
        'label_age': tf.FixedLenFeature([], tf.int64)
        })

    # 將序列化的圖片轉為 uint8 的 tensor
    image = tf.decode_raw(features['image_string'], tf.uint8)

    # 將 label 的資料轉為 float32 的 tensor
    gender_label = tf.cast(features['label_gender'], tf.int64)
    age_label = tf.cast(features['label_age'], tf.int64)

    # 將圖片的大小轉為 int32 的 tensor
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # 將圖片調整成正確的尺寸
    image = tf.reshape(image, [height, width, 3])

    # preprocess
    processed_image = inception_preprocessing.preprocess_image(image, CROP_HEIGHT, CROP_HEIGHT, is_training=False)
    # 這裡可以進行其他的圖形轉換處理 ...
    # ...

    # 圖片的標準尺寸
    #image_size_const = tf.constant((CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.int32)

    # 將圖片調整為標準尺寸
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #target_height=CROP_HEIGHT,
    #target_width=CROP_WIDTH)

    # 打散資料順序
    images, gender_labels, age_labels = tf.train.shuffle_batch(
        [processed_image, gender_label, age_label],
        batch_size=BATCH_SIZE,
        capacity=6400,
        num_threads=1,
        min_after_dequeue=1600)

    return images, gender_labels, age_labels


def train(DATA_SIZE):
    with tf.Graph().as_default():
        #global_stepsss = tf.Variable(0, trainable=False)
        training = tf.placeholder(tf.bool)
        #labels_gender_encode = tf.placeholder(tf.int32, [None, GENDER_CLASS_NUM], name='label_gender-input')
        #labels_age_encode = tf.placeholder(tf.int32, [None, AGE_CLASS_NUM], name='label_age-input')
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=EPOCH)
        # 讀取並解析 TFRecords 的資料
        images, gender_labels, age_labels = read_and_decode(filename_queue)
        with slim.arg_scope(infer.inception_resnet_v2_arg_scope()):
            # 这里如果我们设置num_classes=None,则可以得到restnet输出的瓶颈层，num_classes默认为10001，是用作imagenet的输出层。同样，我们也可以根据需要修改num_classes为其他的值来满足我们的训练要求。
            endpoints = infer.inception_resnet_v2(images, is_training=training)
            print('###############################################################')
            variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'])
            #init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'),slim.get_model_variables('InceptionResnetV2'))
            #init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'),variables_to_restore)

            restorer = tf.train.Saver(variables_to_restore)
        # calculate loss
        # gender
        logit_gender_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=endpoints['LogitsGender'], labels=gender_labels)
        auxlogit_gender_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=endpoints['AuxLogitsGender'], labels=gender_labels)
        logit_gender_loss = tf.reduce_mean(logit_gender_entropy)
        auxlogit_gender_loss = tf.reduce_mean(auxlogit_gender_entropy)
        # age
        logit_age_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=endpoints['LogitsAge'], labels=age_labels)
        auxlogit_age_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=endpoints['AuxLogitsAge'], labels=age_labels)
        logit_age_loss = tf.reduce_mean(logit_age_entropy)
        auxlogit_age_loss = tf.reduce_mean(auxlogit_age_entropy)
        total_loss = (logit_gender_loss + auxlogit_gender_loss + logit_age_loss + auxlogit_age_loss)/4
            
        # Evaluate model
        # gender
        pred_gender = tf.argmax(endpoints['LogitsGender'],1)
        expect_gender = gender_labels
        correct_gender = tf.equal(pred_gender, expect_gender)
        accuracy_gender = tf.reduce_mean(tf.cast(correct_gender, tf.float32))

        # age
        pred_age = tf.argmax(endpoints['LogitsAge'],1)
        expect_age = age_labels
        correct_age = tf.equal(pred_age, expect_age)
        accuracy_age = tf.reduce_mean(tf.cast(correct_age, tf.float32))
            
        # optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = LEARNING_RATE_BASE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:                


            # initial model and variables
            if FIRST_TRAIN :
                print ("Restore  Start!")
                sess.run(init_op)   
                restorer.restore(sess, os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'))                 
                #init_fn(sess)
                print ("Restore  Complete!")
            else :
                print ("Restore  Start!")                    
                sess.run(init_local)
                saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                print ("Restore  Finished!")


            # create thread coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            batch_num = int(DATA_SIZE/BATCH_SIZE)
            step = 1
            while step  <= EPOCH:
                for bp in range(batch_num):                   
                    pred_g, pred_a, expc_g, expc_a, acc_g, acc_a, loss_logit_g, loss_logit_a,\
                        loss_auxlogit_g, loss_auxlogit_a, loss_t, g_step, opt=  \
                        sess.run( [pred_gender, pred_age, expect_gender, expect_age,\
                        accuracy_gender, accuracy_age,logit_gender_loss, logit_age_loss,\
                        auxlogit_gender_loss, auxlogit_age_loss, total_loss, global_step, optimizer],feed_dict={training:True}) 
                    """
                    # show image
                    plt.figure(1) 
                    plt.imshow(batch_images[0])                        
                    plt.show()
                    """
                    if bp % display_step == 0:
                        print("Epoch : " + str(step) +" Iter : " + str((step-1)*batch_num+bp) +" Global Epoch : "+str(int(round(g_step/batch_num)+1))+\
                            "\nTraining Total Loss \t\t= {:.12f}".format(loss_t) +\
                            "\nTraining Gender Logits Loss \t\t= {:.12f}".format(loss_logit_g) +\
                            "\nTraining Gender AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_g) +\
                            "\nTraining Age Logits Loss \t\t= {:.12f}".format(loss_logit_a) +\
                            "\nTraining Age AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_a))

                        print("Gender ACC \t: ",acc_g)
                        print("Gender Expected Value \t: ",expc_g)   
                        print("Gender Predict Value \t: ",pred_g)

                        print("Age ACC \t: ",acc_a)
                        print("Age Expected Value \t: ",expc_a)   
                        print("Age Predict Value \t: ",pred_a)
                save_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, save_path)
                step += 1
            # close thread queue
            coord.request_stop()
            coord.join(threads)
    return
def main(argv=None):
    DATA_SIZE = get_data_size()
    print("Training Start")
    train(DATA_SIZE)
    print('Training Done')
if __name__ == '__main__':
    main()



