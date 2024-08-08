#!/bin/bash

# Set default values for hyperparameters

weights="Your trained weights" # Path to your trained weights
infer_list="data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt" # Path to the list of images for inference
num_workers=10 # Number of workers for data loading
num_classes=21 # Number of classes
tblog="" # Path to the TensorBoard log directory (optional)
voc12_root="data/VOCdevkit/VOC2012" # Root path for VOC2012 dataset
out_npy="Path where npy is saved" # Path where the output .npy files will be saved
out_png="" # Path where the output .png files will be saved (optional)

# Run the inference script with the specified hyperparameters
python3 infer.py \
  --weights $weights \
  --infer_list $infer_list \
  --num_workers $num_workers \
  --num_classes $num_classes \
  --tblog $tblog \
  --voc12_root $voc12_root \
  --out_npy $out_npy \
  --out_png $out_png
