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
bash Run_train.sh
```

## CAM generation
To generate CAM using the model, please follow these steps:

### Step 1: Modify run_infer.sh
Edit the run_infer.sh script to set the appropriate local paths for your environment.

### Step 2: Configure Hyperparameters
You can use the provided hyperparameters or adjust them according to your needs.

### Step 3: Run the Inference Script
After modifying the script and configuring the hyperparameters, run the inference script using the following command:
```
bash run_infer.sh
```

## CAM refinement & Pseudo label generation
Completing PLmix Training and Inference

We have successfully enhanced the quality of CAMs in the benchmark using PLmix by following these steps. For subsequent steps, including Stage 2 (Refine CAM) and Stage 3 (Segmentation Model Training), please refer to the [MuSCLe benchmark](https://github.com/SCoulY/MuSCLe?tab=readme-ov-file).

We thank MuSCLe for providing the code and framework for these subsequent stages.
