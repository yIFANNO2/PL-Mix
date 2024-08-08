# PL-Mix
The implementation of: Refining Pseudo-labels through Iterative Mix-Up for Weakly Supervised Semantic Segmentation

## Preparation

### Data
- Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). Put the data under ./data/VOC2012 folder.

### Model
- You can download our pre-trained model weights from [google drive](https://drive.google.com/drive/folders/1E1gweNZWHyAJ47cxupf4R1YV_j8hZN1-?usp=sharing) for immediate testing. Alternatively, you can follow the tutorial below to train the model from scratch.


### Packages
- install conda from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- build cond env from file
```
conda env create --name PLmix --file environment.yaml
conda activate PLmix
```
## MCL training
To train the model, please follow these steps:
### Step 1: Modify run_train.sh
Edit the run_train.sh script to set the appropriate local paths for your environment.
### Step 2: Configure Hyperparameters
You can use the provided hyperparameters or adjust them according to your needs
### Step 3: Run the Training Script
After modifying the script and configuring the hyperparameters, run the training script using the following command:
```
bash run_train.sh
```

## CAM generation
### specify dir ```--tblog XXX/tblog ``` to show raw CAM visualisation in tensorboard
```
python infer_mcl.py --voc12_root PATH_TO_VOC12 --infer_list PATH_TO_INFER_LIST --weights PATH_TO_TRAINED_MODEL --out_npy OUTPUT_DIR
```

## CAM refinement & Pseudo label generation
### turn on flag ```--soft_output 1 ``` to store soft pseudo labels for BEACON training
```
python infer_irn.py --cam_dir CAM_DIR --sem_seg_out_dir OUTPUT_PSEUDO_LABEL_DIR --soft_output 0 --irn_weights_name PATH_TO_PRETRAINED_IRN_MODEL
```

## CAM quality evaluation
### Raw CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type npy --list data/train.txt --predict_dir CAM_DIR --curve True
cd ..
```

### Refined CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type png --list data/train.txt --predict_dir REFINED_CAM_DIR 
cd ..

### Statistical Significance
```
cd src
python evaluation.py --comment COMMENTS --type png --list data/train.txt --predict_dir REFINED_CAM_DIR 
cd ..
