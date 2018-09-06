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
import res_inference as infer
import Data_Loader_nctu as ld
import Label_Encoder as le
import Data_Augmentation as dt_aug
import MinMax_Scaler as mms
import time
import gc 
import skimage.io as io
# Parameters

# TFRecords 檔案名稱
tfrecords_filename = 'test.tfrecords'

DATA_NAME = "tfr"
DATA_PATH = "../Data/"+DATA_NAME
MODEL_SAVE_PATH = "./"+DATA_NAME+"/"
MODEL_NAME = "ResNet50_model"
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.995



# Train setting
BATCH_SIZE = 32
MIN_KM = 5
EPOCH = 100
save_step = 20
DECAY_STEPS = 100*BATCH_SIZE
# Network Parameters
GENDER_CLASS_NUM = 2
AGE_CLASS_NUM = 3
n_input = 50176

# image setting
CROP_HEIGHT= 224
CROP_WIDTH = 224
NUM_CHANNELS = 3

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

    # 這裡可以進行其他的圖形轉換處理 ...
    # ...

    # 圖片的標準尺寸
    #image_size_const = tf.constant((CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.int32)

    # 將圖片調整為標準尺寸
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    target_height=CROP_HEIGHT,
    target_width=CROP_WIDTH)

    # 打散資料順序
    images, gender_labels, age_labels = tf.train.shuffle_batch(
        [resized_image, gender_label, age_label],
        batch_size=BATCH_SIZE,
        capacity=6400,
        num_threads=1,
        min_after_dequeue=800)

    return images, gender_labels, age_labels


def test(DATA_SIZE):
    with tf.Session() as sess: 
        # define the placeholder
        training = tf.placeholder(tf.bool)
        image_ = tf.placeholder(tf.float32, [None, n_input],name='image-input')
        image = tf.reshape(image_, shape=[-1, CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS])

            
        label_gender_encode = tf.placeholder(tf.int32, [None, GENDER_CLASS_NUM], name='label_gender-input')
        label_age_encode = tf.placeholder(tf.int32, [None, AGE_CLASS_NUM], name='label_age-input')
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
            
        # Construct model
        pred_gender_encode, pred_age_encode = infer.inference(image,  training)

        # Define global step
        global_step = tf.Variable(0, trainable=False)

        # Evaluate model
        pred_gender = tf.argmax(pred_gender_encode,1)
        pred_age = tf.argmax(pred_age_encode,1)
        expect_gender = tf.argmax(label_gender_encode,1)
        expect_age = tf.argmax(label_age_encode,1)
        correct_gender = tf.equal(pred_gender, expect_gender)
        correct_age = tf.equal(pred_age, expect_age)
        accuracy_gender = tf.reduce_mean(tf.cast(correct_gender, tf.float32))
        accuracy_age = tf.reduce_mean(tf.cast(correct_age, tf.float32))
        #Define learning rate
        learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE_BASE,global_step = global_step,
            decay_steps = DATA_SIZE*BATCH_SIZE, decay_rate = LEARNING_RATE_DECAY, staircase = True, name = None)
        #learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, batch_num, LEARNING_RATE_DECAY, staircase=True)
        # cycle=True
        #learing_rate2 = tf.train.polynomial_decay(
        #    learning_rate=0.1, global_step=global_step, decay_steps=50,
        #    end_learning_rate=0.01, power=0.5, cycle=True)
        # Define loss and optimizer
        #cost_gender = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_gender_encode, labels=label_gender_encode))
        #cost_age = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_age_encode, labels=label_age_encode))
        cross_entropy_gender = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_gender_encode, labels=label_gender_encode)
        cross_entropy_age = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_age_encode, labels=label_age_encode)        
        #count = tf.constant(0)
            
        # loss function
        def loss_function(cross_entropy, expected_labels, class_num):
            # size -> input batch size
            # MK -> each class's num
            # NK -> 1/MK
            # NKS -> sum of NK
            # BK -> weight of each class's loss
            # BK_L -> BK Limited by BK_MAX
            # BK_W -> BK weight list for loss            
            size = tf.shape(expected_labels)[0]
            MK = tf.bincount(tf.cast(expected_labels, tf.int32), minlength = class_num)
            MK_masked = tf.less(MK,MIN_KM)
            MK_L = tf.where(MK_masked, tf.fill([class_num],MIN_KM), MK)
            NK = tf.divide(1,tf.cast(MK_L, dtype=tf.float32))
            NKS = tf.reduce_sum(NK)
            BK = tf.divide(NK,NKS)
                
            BK_W = tf.zeros(size, dtype=tf.float32, name=None)
            count = 0
            def BK_set(count, expected_labels, class_num, BK, BK_W): 
                size = tf.shape(expected_labels)[0]
                zeros = tf.zeros(size, dtype=tf.float32, name="zeros")  
                masked = tf.equal(expected_labels, tf.cast(count,dtype=tf.int64))                
                temp_BK = tf.where(masked,  tf.fill([size], BK[count]), zeros)
                BK_W = tf.add(BK_W, temp_BK)
                count = count + 1
                return count, expected_labels, class_num, BK, BK_W                
            count, expected_labels, class_num, BK, BK_W = tf.while_loop((lambda count, expected_labels, class_num, BK_L, BK_W: tf.less(count, class_num)), BK_set, [count, expected_labels, class_num, BK, BK_W])

            weighted_cross_entropy = tf.multiply(cross_entropy, BK_W)
            cost = tf.reduce_mean(weighted_cross_entropy)            
            return cost
            
        # get cost
        cost_gender = loss_function(cross_entropy_gender, expect_gender, GENDER_CLASS_NUM)
        cost_age = loss_function(cross_entropy_age, expect_age, AGE_CLASS_NUM)       
            
        # choose cost age or cost gender by turns
        #cost = tf.cond(tf.equal(global_step%2,0), lambda: cost_age , lambda: cost_gender)
        # average cost
        cost = (cost_age + cost_gender)/2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #child_optimizer = tf.contrib.layers.optimize_loss(child_cost, child_global_step, child_learning_rate, optimizer='Adam')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
            
        saver = tf.train.Saver(var_list=tf.global_variables())     


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
        # 讀取並解析 TFRecords 的資料
        images, gender_labels, age_labels = read_and_decode(filename_queue)
        #init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        print ("Restore  Start!")
        saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        sess.run(init_local)
        print ("Restore  Finished!")

        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)

        BATCH_NUM = int(DATA_SIZE/BATCH_SIZE)
        #开始一个epoch的训练
        steps = 0
        confusion_matrix_gender = np.zeros([GENDER_CLASS_NUM,GENDER_CLASS_NUM], dtype=float)        
        confusion_matrix_age = np.zeros([AGE_CLASS_NUM,AGE_CLASS_NUM], dtype=float)      
        for step in range(BATCH_NUM):
            steps += 1;
            #print("get data")
            batch_images, batch_gender_labels, batch_age_labels = sess.run([images, gender_labels, age_labels])
            # 檢查每個 batch 的圖片維度
            #print(img.shape)
            #print("process data")
            test_batch_labels_gender_encode = le.encode_labels(batch_gender_labels, GENDER_CLASS_NUM)
            test_batch_labels_age_encode = le.encode_labels(batch_age_labels, AGE_CLASS_NUM)

            test_pred_gender, test_pred_age,            \
                test_cost_gender, test_cost_age,        \
                test_acc_gender, test_acc_age =         \
                sess.run([ pred_gender, pred_age, \
                cost_gender, cost_age,                          \
                accuracy_gender, accuracy_age],                 \
                feed_dict={training:False,                       \
                image: batch_images,               \
                label_gender_encode: test_batch_labels_gender_encode, \
                label_age_encode: test_batch_labels_age_encode})  
            for ci in range(len(batch_gender_labels)):
                confusion_matrix_gender[batch_gender_labels[ci],test_pred_gender[ci]] +=1;
            for ci in range(len(batch_age_labels)):
                confusion_matrix_age[batch_age_labels[ci],test_pred_age[ci]] +=1;
                    #print(al.shape)
        print_confusion_matrix(confusion_matrix_gender)
        print_confusion_matrix(confusion_matrix_age)
        
        coord.request_stop()
        coord.join(threads)


        # 顯示每個 batch 的第一張圖
        #io.imshow(img[0, :, :, :])
        #plt.show()
    return
      
def main(argv=None):
    DATA_SIZE = get_data_size()
    print("Testing Start")
    test(DATA_SIZE)
    print("Testing Done")
if __name__ == '__main__':
    main()



