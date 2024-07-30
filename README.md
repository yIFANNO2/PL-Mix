# PL-Mix
The implementation of: Refining Pseudo-labels through Iterative Mix-Up for Weakly Supervised Semantic Segmentation

## Preparation

### Data
- Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). Put the data under ./data/VOC2012 folder.

### Model
- Download model weights from [google drive](https://drive.google.com/drive/folders/1K3mMECLdWdu8YVrMq8YblppRdLtCcAaW?usp=sharing), including pretrained MCL, MuSCLe and IRN models.

### Packages
- install conda from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- build cond env from file
```
conda env create --name muscle --file environment.yaml
conda activate muscle
```
