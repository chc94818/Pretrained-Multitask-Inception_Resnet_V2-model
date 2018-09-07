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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TFRecords 檔案名稱
tfrecords_filename = 'train.tfrecords'

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
flags.DEFINE_string('checkpoints_dir', '/Multi-Task_CNN/pretrain_model/checkpoints/' ,'Directory where checkpoints are saved.')
flags.DEFINE_string('trained_checkpoints_dir', '/Multi-Task_CNN/pretrain_model/fine_tuned_model/','Directory where checkpoints and event logs are written to.')
flags.DEFINE_string('model_name', 'inception_resnet_v2.ckpt', 'The name of the architecture to train.')
flags.DEFINE_string('checkpoint_exclude_scopes', None, 'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes', None,'Comma-separated list of scopes to filter the set of variables to train. By default, None would train all the variables.')
flags.DEFINE_integer('epochs', 5, 'The num of training epochs.')
flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('dispaly_every_n_steps', 20, 'The frequency with which logs are print.')
flags.DEFINE_integer('save_every_n_steps', 100, 'The frequency with which model is saved.')

FLAGS = flags.FLAGS
FLAGS(sys.argv) 

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
    processed_image = inception_preprocessing.preprocess_image(image, CROP_HEIGHT, CROP_HEIGHT, is_training=True)
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

# function get trainable scope
def get_trainable_scopes():
    trainable_scopes = []
    if FLAGS.trainable_scopes:
        trainable_scopes = [scope.strip()for scope in FLAGS.trainable_scopes.split(',')]
        scopes = [scope.strip() for scope in trainable_scopes]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)        
        print('train %s layers' % (scopes))
        print('###############################################################')
        return variables_to_train
    else:        
        print('train all layers')
        print('###############################################################')
        return tf.trainable_variables()    
    

# function get restore scope
def get_restore_variables():
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    #print(exclusions)
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)
    return variables_to_restore

# function train
def train(DATA_SIZE):
    with tf.Graph().as_default():
        #global_stepsss = tf.Variable(0, trainable=False)
        training = tf.placeholder(tf.bool)
        #labels_gender_encode = tf.placeholder(tf.int32, [None, GENDER_CLASS_NUM], name='label_gender-input')
        #labels_age_encode = tf.placeholder(tf.int32, [None, AGE_CLASS_NUM], name='label_age-input')
        filename_queue = tf.train.string_input_producer([tfrecords_filename])
        # 讀取並解析 TFRecords 的資料
        images, gender_labels, age_labels = read_and_decode(filename_queue)
        with slim.arg_scope(infer.inception_resnet_v2_arg_scope()):
            # 这里如果我们设置num_classes=None,则可以得到restnet输出的瓶颈层，num_classes默认为10001，是用作imagenet的输出层。同样，我们也可以根据需要修改num_classes为其他的值来满足我们的训练要求。
            endpoints = infer.inception_resnet_v2(images, is_training=training)
            ##################################################################################
            variables_to_restore = get_restore_variables()
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
        learning_rate = FLAGS.learning_rate


        # get trainable variable
        variables_to_train = get_trainable_scopes()
        #variables_to_train = get_trainable_scopes(['InceptionResnetV2/AuxLogits'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # optimizer
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, var_list=variables_to_train, global_step=global_step)

        # initial operation
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())            
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:                
            
            # initial model and variables
            if os.path.join(FLAGS.checkpoints_dir, FLAGS.model_name) == checkpoints_path :
                print ("Restore pretrained Start!")
                sess.run(init_op)   
                restorer.restore(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_name))                 
                #init_fn(sess)
                print ("Restore  Complete!")
            elif FLAGS.trained_checkpoints_dir == FLAGS.checkpoints_dir:
                print ("Restore continue fine-tuned Start!")                    
                sess.run(init_op)
                saver.restore(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_name))
                print ("Restore  Finished!")
            else:
                print ("Restore fine-tuned and create new checkpoints Start!")
                sess.run(init_op)   
                restorer.restore(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_name))                 
                #init_fn(sess)
                print ("Restore  Complete!")

            # create thread coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # restore global step to enter training step
            g_step = sess.run(global_step) 
            batch_num_per_epoch = int(DATA_SIZE/FLAGS.batch_size)
            batch_num_total = batch_num_per_epoch * FLAGS.epochs          
            print('Total Steps : ', batch_num_total)    
            print('Start Gloabal Steps : ', g_step)
            print('###############################################################')
            while g_step <= batch_num_total:
                # training session     
                pred_g, pred_a, expc_g, expc_a, acc_g, acc_a, loss_logit_g, loss_logit_a,\
                    loss_auxlogit_g, loss_auxlogit_a, loss_t, g_step, opt=  \
                    sess.run( [pred_gender, pred_age, expect_gender, expect_age,\
                    accuracy_gender, accuracy_age,logit_gender_loss, logit_age_loss,\
                    auxlogit_gender_loss, auxlogit_age_loss, total_loss, global_step, optimizer],feed_dict={training:True}) 
                #print("vtt, ", vtt)

                # display training status
                if g_step % FLAGS.dispaly_every_n_steps == 0 or g_step== batch_num_total:
                    print("Epoch : " + str(int((g_step)/batch_num_per_epoch)) +"/"+str(FLAGS.epochs)+" Iter : " + str(g_step)+"/"+str(batch_num_total)+\
                        "\nTraining Total Loss \t\t= {:.12f}".format(loss_t) +\
                        "\nTraining Gender Logits Loss \t= {:.12f}".format(loss_logit_g) +\
                        "\nTraining Gender AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_g) +\
                        "\nTraining Age Logits Loss \t= {:.12f}".format(loss_logit_a) +\
                        "\nTraining Age AuxLogits Loss \t= {:.12f}".format(loss_auxlogit_a))

                    print("Gender ACC \t: ",acc_g)
                    print("Gender Expected Value \t: ",expc_g)   
                    print("Gender Predict Value \t: ",pred_g)

                    print("Age ACC \t: ",acc_a)
                    print("Age Expected Value \t: ",expc_a)   
                    print("Age Predict Value \t: ",pred_a)
                    print('###############################################################')
                # middle-term save model
                if g_step % FLAGS.save_every_n_steps == 0:
                    print("Step Save Strart, Iter : ", str(g_step))
                    saver.save(sess, os.path.join(FLAGS.trained_checkpoints_dir, FLAGS.model_name))
                    print("Step Save Complete")
                    print('###############################################################')
            
            # final-term save model
            print("Final Save Strart")
            saver.save(sess, os.path.join(FLAGS.trained_checkpoints_dir, FLAGS.model_name))
            print("Final Save Complete")
            print('###############################################################')
            # close thread queue
            coord.request_stop()
            coord.join(threads)
    return
def main(argv=None):
    # get total data size
    DATA_SIZE = get_data_size()
    print('###############################################################')
    print('#####################TRAIN_SESSION_START#######################')
    print('###############################################################')
    # display flags args
    print('checkpoints_dir\t\t\t: ', FLAGS.checkpoints_dir)
    print('trained_checkpoints_dir\t\t: ', FLAGS.trained_checkpoints_dir)
    print('model_name\t\t\t: ', FLAGS.model_name)
    print('checkpoint_exclude_scopes\t: ', FLAGS.checkpoint_exclude_scopes)
    print('trainable_scopes\t\t: ', FLAGS.trainable_scopes)
    print('epochs\t\t\t\t: ', FLAGS.epochs)
    print('batch_size\t\t\t: ', FLAGS.batch_size)
    print('learning_rate\t\t\t: ', FLAGS.learning_rate)
    print('dispaly_every_n_steps\t\t: ', FLAGS.dispaly_every_n_steps)
    print('save_every_n_steps\t\t: ', FLAGS.save_every_n_steps)

    # training start
    #print("Training Start")
    train(DATA_SIZE)
    #print('Training Done')
    print('###############################################################')
    print('##################TRAIN_SESSION_COMPLETE#######################')
    print('###############################################################')
if __name__ == '__main__':
    main()



