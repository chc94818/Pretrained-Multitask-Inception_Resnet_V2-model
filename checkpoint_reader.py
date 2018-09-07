from tensorflow.python import pywrap_tensorflow
import os

MODEL_SAVE_PATH = './fine_tuned_model/'
MODEL_BACKUP_SAVE_PATH = './fine_tuned_model_backup/'
MODEL_NAME = 'inception_resnet_v2.ckpt'
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.995

checkpoints_dir = '/Multi-Task_CNN/pretrain_model/checkpoints/'
checkpoints_filename = 'inception_resnet_v2.ckpt'


checkpoint_path = os.path.join(MODEL_SAVE_PATH, checkpoints_filename)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

print('MODEL_SAVE_PATH')
for key in var_to_shape_map:
	if key.find('InceptionResnetV2/Repeat_2/block8_4/Branch_1/Conv2d_0c_3x1/weights') != -1:
		print("tensor_name: ", key)
		print(reader.get_tensor(key))
	if key.find('InceptionResnetV2/Logits/Age/Logits/weights') != -1:
		print("tensor_name: ", key)
		print(reader.get_tensor(key))
	if key.find('InceptionResnetV2/AuxLogits/Age/Logits/') != -1:
		print("tensor_name: ", key)
		print(reader.get_tensor(key))




