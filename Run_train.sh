#!/bin/bash

# Set default values for hyperparameters

ablation_name="YourProjectName" # Name of your project
batch_size=12 # Size of the training batch
max_epoches=30 # Maximum number of training epochs
lr=3e-5 # Learning rate
num_workers=8 # Number of workers for data loading
wt_dec=1e-6 # Weight decay
train_list="voc12/train_aug.txt" # Path to the training data list
val_list="data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt" # Path to the validation data list
gt="data/VOCdevkit/VOC2012/ImageSets/Segmentation" # Path to the ground truth data
num_classes=21 # Number of classes
base_path="YourProjectPath" # Base path for the project
session_name="${base_path}/${ablation_name}/EffiB3_PLMix" # Session name for saving checkpoints
crop_size=448 # Crop size for input images
weights=None # Path to the pretrained weights
voc12_root="${base_path}/data/VOCdevkit/VOC2012" # Root path for VOC2012 dataset
tblog_dir="${base_path}/${ablation_name}/tblog_new" # Directory for tensorboard logs
seed=3407 # Random seed for reproducibility
Random_mix=True # Use random mix of data
Random_order=True # Use random order of data
mix_cls_loss=False # Use mixed class loss
crf=True # Use Conditional Random Field (CRF) post-processing
mix_loss="Focal_loss" # Type of loss function to use (Focal_loss, CE_loss, Focal_CE_loss)
test_path="${base_path}/${ablation_name}_npy" # Path to save test results
test_path_png="${base_path}/${ablation_name}_png" # Path to save test results in PNG format
log_name="${base_path}/${ablation_name}_log" # Path to save log files
camT=0.30 # Threshold for Class Activation Maps (CAM)
MaxmIou=50 # Maximum IoU threshold

# Run the training script with the specified hyperparameters
python train_Muscle_PLMIX.py \
  --ablation_name $ablation_name \
  --batch_size $batch_size \
  --max_epoches $max_epoches \
  --lr $lr \
  --num_workers $num_workers \
  --wt_dec $wt_dec \
  --train_list $train_list \
  --val_list $val_list \
  --gt $gt \
  --num_classes $num_classes \
  --session_name $session_name \
  --crop_size $crop_size \
  --weights $weights \
  --voc12_root $voc12_root \
  --tblog_dir $tblog_dir \
  --seed $seed \
  --Random_mix $Random_mix \
  --Random_order $Random_order \
  --mix_cls_loss $mix_cls_loss \
  --crf $crf \
  --mix_loss $mix_loss \
  --test_path $test_path \
  --test_path_png $test_path_png \
  --log_name $log_name \
  --camT $camT \
  --MaxmIou $MaxmIou
