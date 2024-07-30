# PL-Mix
The implementation of: Refining Pseudo-labels through Iterative Mix-Up for Weakly Supervised Semantic Segmentation

## Preparation

### Data
- Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). Put the data under ./data/VOC2012 folder.

### Model
- You can download our pre-trained model weights from Google Drive for immediate testing. Alternatively, you can follow the tutorial below to train the model from scratch.

### Packages
- install conda from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- build cond env from file
```
conda env create --name muscle --file environment.yaml
conda activate muscle
```
