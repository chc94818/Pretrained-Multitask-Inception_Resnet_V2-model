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
import time
import gc 
import skimage.io as io
from absl import app
from absl import flags
# Parameters

# TFRecords 檔案名稱
tfrecords_filename = 'test.tfrecords'

train_path_dir= '/Multi-Task_CNN/pretrain_model/fine_tuned_model/inception_resnet_v2.ckpt'
checkpoints_path = '/Multi-Task_CNN/pretrain_model/checkpoints/inception_resnet_v2.ckpt'
model_name = 'InceptionResnetV2'

# Network Parameters
GENDER_CLASS_NUM = 2
AGE_CLASS_NUM = 3

# image setting
CROP_HEIGHT= 299
CROP_WIDTH = 299

slim = tf.contrib.slim

# Train setting
flags.DEFINE_string('checkpoints_dir', '/Multi-Task_CNN/pretrain_model/fine_tuned_model/' ,'Directory where checkpoints are saved.')
flags.DEFINE_string('model_name', 'inception_resnet_v2.ckpt', 'The name of the architecture to train.')
flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')
flags.DEFINE_integer('dispaly_every_n_steps', 20, 'The frequency with which logs are print.')

FLAGS = flags.FLAGS
FLAGS(sys.argv) 

# funtion print confusion matrix
def print_confusion_matrix(confusion_matrix):
    # get confusion matrix
    # temp_sc -> sum of correct
    # temp_s -> sum
    # temp_dp -> denominator of precision
    # temp_dr -> denominator of recall
    # temp_p -> precision
    # temp_r -> recall

    class_num = confusion_matrix.shape[0]
    precision = np.zeros([class_num], dtype=float)
    recall = np.zeros([class_num], dtype=float)
    accuracy = 0
    temp_sc = 0
    temp_s = 0
    for cfm_g_r in range(0,class_num) :
        temp_sc = temp_sc +  confusion_matrix[cfm_g_r, cfm_g_r]
        temp_dp = 0
        temp_dr = 0
        for cfm_g_c in range(0,class_num) :                
            temp_dp = temp_dp + confusion_matrix[cfm_g_c, cfm_g_r]
            temp_dr = temp_dr + confusion_matrix[cfm_g_r, cfm_g_c]
        temp_s = temp_s + temp_dp
        if(temp_dp == 0):
            temp_p = 0
        else :
            temp_p = confusion_matrix[cfm_g_r, cfm_g_r]/temp_dp
        if(temp_dr == 0):
            temp_r = 0
        else :
            temp_r = confusion_matrix[cfm_g_r, cfm_g_r]/temp_dr
        
        precision[cfm_g_r] = temp_p
        recall[cfm_g_r] = temp_r
    accuracy = temp_sc/temp_s
    ap = sum(precision) / float(len(precision))

    print("Confusion Matrix : ")
    print("Matrix\t:\t", end = '')
    for ci in range(0,class_num) :
        print("p%d\t" % (ci+1), end = '')
    print("R")
    for ri in range(0,class_num) :
        print("e%d\t:\t" % (ri+1), end = '')
        for ci in range(0,class_num) :
            print("%-6d\t" % (confusion_matrix[ri,ci]), end = '')
        print("%.2f%%" % (recall[ri]*100))
    print("P\t:\t", end = '')
    for ci in range(0,class_num) :
        print("%.2f%%\t" % (precision[ci]*100), end = '')
    print("%.2f%%\t" % (accuracy*100), end = '')
    print("%.2f%%" % (ap*100))
        
    return
# function get total data size
def get_data_size():
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    counter = 0
    for string_record in record_iterator:
       counter += 1
    return counter

# function get batch data
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
        batch_size=FLAGS.batch_size,
        capacity=640,
        num_threads=1,
        min_after_dequeue=320)

    return images, gender_labels, age_labels

# function test
def test(DATA_SIZE):
    with tf.Graph().as_default():
        #global_stepsss = tf.Variable(0, trainable=False)
        training = tf.placeholder(tf.bool)
        #labels_gender_encode = tf.placeholder(tf.int32, [None, GENDER_CLASS_NUM], name='label_gender-input')
        #labels_age_encode = tf.placeholder(tf.int32, [None, AGE_CLASS_NUM], name='label_age-input')
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
        # 讀取並解析 TFRecords 的資料
        images, gender_labels, age_labels = read_and_decode(filename_queue)
        with slim.arg_scope(infer.inception_resnet_v2_arg_scope()):
            # 这里如果我们设置num_classes=None,则可以得到restnet输出的瓶颈层，num_classes默认为10001，是用作imagenet的输出层。同样，我们也可以根据需要修改num_classes为其他的值来满足我们的训练要求。
            endpoints = infer.inception_resnet_v2(images, is_training=training)
            print('###############################################################')
            variables_to_restore = slim.get_variables_to_restore()
            #init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'),slim.get_model_variables('InceptionResnetV2'))
            #init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'),variables_to_restore)
            # model restorer
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
        #total_loss = (logit_gender_loss + auxlogit_gender_loss + logit_age_loss + auxlogit_age_loss)/4
        total_loss = tf.reduce_mean([logit_gender_loss, auxlogit_gender_loss, logit_age_loss, auxlogit_age_loss])
            
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
        #learning_rate = LEARNING_RATE_BASE
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:                
            print('###############################################################')
            print('#####################TEST_SESSION_START########################')
            print('###############################################################')

            # restore model and variables    
            print ("Restore fine-tuned Start!")                    
            sess.run(init_op)   
            restorer.restore(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_name))                 
            print ("Restore  Finished!")


            # create thread coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # get batch num per epoch
            batch_num_per_epoch = int(DATA_SIZE/FLAGS.batch_size)
            # confusion matrix
            confusion_matrix_gender = np.zeros([GENDER_CLASS_NUM,GENDER_CLASS_NUM], dtype=float)        
            confusion_matrix_age = np.zeros([AGE_CLASS_NUM,AGE_CLASS_NUM], dtype=float)

            # test all batch in test dataset
            for bp in range(batch_num_per_epoch): 
                # test session         
                pred_g, pred_a, expc_g, expc_a, acc_g, acc_a, loss_logit_g, loss_logit_a,\
                    loss_auxlogit_g, loss_auxlogit_a, loss_t=  \
                    sess.run( [pred_gender, pred_age, expect_gender, expect_age,\
                    accuracy_gender, accuracy_age,logit_gender_loss, logit_age_loss,\
                    auxlogit_gender_loss, auxlogit_age_loss, total_loss],feed_dict={training:False}) 
                """
                # show image
                plt.figure(1) 
                plt.imshow(batch_images[0])                        
                plt.show()
                """
                # set confusion matrix
                # gender
                for ci in range(len(expc_a)):
                    confusion_matrix_gender[expc_g[ci],pred_g[ci]] +=1;
                # age 
                for ci in range(len(expc_a)):
                    confusion_matrix_age[expc_a[ci],pred_a[ci]] +=1;

                # display middle-term test status
                if bp % FLAGS.dispaly_every_n_steps == 0:
                    print("Iter : " + str(bp+1) +\
                        "\nTesting Total Loss \t\t= {:.12f}".format(loss_t) +\
                        "\nTesting Gender Logits Loss \t\t= {:.12f}".format(loss_logit_g) +\
                        "\nTesting Gender AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_g) +\
                        "\nTesting Age Logits Loss \t\t= {:.12f}".format(loss_logit_a) +\
                        "\nTesting Age AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_a))

                    print("Gender ACC \t: ",acc_g)
                    print("Gender Expected Value \t: ",expc_g)   
                    print("Gender Predict Value \t: ",pred_g)

                    print("Age ACC \t: ",acc_a)
                    print("Age Expected Value \t: ",expc_a)   
                    print("Age Predict Value \t: ",pred_a)
            # close thread queue
            coord.request_stop()
            coord.join(threads)
    return confusion_matrix_gender, confusion_matrix_age,
def main(argv=None):
    # get total data size
    DATA_SIZE = get_data_size()

    # display flags args
    print('checkpoints_dir\t\t\t: ', FLAGS.checkpoints_dir)
    print('model_name\t\t\t: ', FLAGS.model_name)
    print('batch_size\t\t\t: ', FLAGS.batch_size)
    print('dispaly_every_n_steps\t\t: ', FLAGS.dispaly_every_n_steps)

    print("Testing Start")
    test_confusion_matrix_gender, test_confusion_matrix_age = test(DATA_SIZE)
    print('Testing Done')
    print("Test Confusion Matrix of Gender")
    print_confusion_matrix(test_confusion_matrix_gender)
    print("Test Confusion Matrix of Age")
    print_confusion_matrix(test_confusion_matrix_age)
if __name__ == '__main__':
    main()



