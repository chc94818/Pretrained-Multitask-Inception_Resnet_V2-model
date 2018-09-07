# This script performs the following operations:
# 1. train multitask gender and age model
# 2. set checkpoints_dir == trained_checkpoints_dir is the 'Continue Training Mode'
# ,that will make checkpoint_exclude_scopes be ignored.
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/Multi-Task_CNN/pretrain_model/checkpoints/


# Where the fine-tuned Inception Resnet V2 checkpoint is saved to.
FINE_TUNED_CHECKPOINT_DIR=/Multi-Task_CNN/pretrain_model/fine_tuned_model/

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAINED_CHECKPOINT_DIR=/Multi-Task_CNN/pretrain_model/fine_tuned_model/

# Where the first fine-tuned FC Inception Resnet V2 checkpoint is saved to.
FC_CHECKPOINT_DIR=/Multi-Task_CNN/pretrain_model/fine_tuned_fc_model/

# Where the first fine-tuned FC Inception Resnet V2 checkpoint is saved to.
ALL_CHECKPOINT_DIR=/Multi-Task_CNN/pretrain_model/fine_tuned_all_model/

# Model's name
MODEL_NAME=inception_resnet_v2.ckpt


# Fine-tune only the fc layers with pretrained model for 50 epochs.
python train.py \
  --checkpoints_dir=${PRETRAINED_CHECKPOINT_DIR}\
  --trained_checkpoints_dir=${FC_CHECKPOINT_DIR} \
  --model_name=${MODEL_NAME}\
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --epochs=50\
  --batch_size=16 \
  --learning_rate=0.01 \
  --dispaly_every_n_steps=100 \
  --save_every_n_steps=500


# Test fine-tuned model
python evaluate.py \
  --checkpoints_dir=${FC_CHECKPOINT_DIR}\
  --model_name=${MODEL_NAME}\
  --batch_size=10 \
  --dispaly_every_n_steps=100


# Fine-tune all layers with pretrained model for 100 epochs.
python train.py \
  --checkpoints_dir=${FC_CHECKPOINT_DIR}\
  --trained_checkpoints_dir=${ALL_CHECKPOINT_DIR} \
  --model_name=${MODEL_NAME}\
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --epochs=100\
  --batch_size=16 \
  --learning_rate=0.0001 \
  --dispaly_every_n_steps=100 \
  --save_every_n_steps=500


# Test fine-tuned model
python evaluate.py \
  --checkpoints_dir=${ALL_CHECKPOINT_DIR}\
  --model_name=${MODEL_NAME}\
  --batch_size=10 \
  --dispaly_every_n_steps=100

  # Fine-tune all layers with pretrained model for 200 epochs.
python train.py \
  --checkpoints_dir=${ALL_CHECKPOINT_DIR}\
  --trained_checkpoints_dir=${ALL_CHECKPOINT_DIR} \
  --model_name=${MODEL_NAME}\
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --epochs=200\
  --batch_size=16 \
  --learning_rate=0.0001 \
  --dispaly_every_n_steps=100 \
  --save_every_n_steps=500


# Test fine-tuned model
python evaluate.py \
  --checkpoints_dir=${ALL_CHECKPOINT_DIR}\
  --model_name=${MODEL_NAME}\
  --batch_size=10 \
  --dispaly_every_n_steps=100