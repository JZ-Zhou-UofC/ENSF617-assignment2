
---
# Assignment 2 Group 5
# Multimodal Garbage Classification Project (Image + Text)

## Overview

This project implements a multimodal classification system that combines both image and text features to improve classification performance. The image model and text model were trained separately, and their outputs are combined in a final notebook to produce the final predictions.

The workflow consists of three main components:

* Text model training (Jupitor Notebook)
* Image model training (TALC cluster)
* Confidence based selection combining both models (Notebook)

---

## Components

### 1. Text Model (Trained locally)

Location:

```text
./text_model.ipynb
```

Description:

* The tutorial text model training is slightly modified to produce a better result
* Transfer learning is used with a pretrained CNN architecture.
* DistilBertModel is used as a base model.
* Checkpoints are saved during training.
* Best accuracy: 85.31%


### 2. Image Model (Trained on TALC Cluster)

training script:

```text
./best_Image_classification_model.py
```
slurm file to run the script on TALC:

```text
./slurm
```
Description:

* The image model is trained on the TALC high-performance computing cluster.
* Images are loaded using PyTorch ImageFolder.
* Transfer learning is used with a pretrained CNN architecture.
* EfficientNet.b2 is choosen to be the based model
* Checkpoints are saved during training.
* Best accuracy: 78.5%


Test training Location:
```text
./testing_folder
```

### 3. Confidence based selection

Location:

```text
confidence_based_selection.ipynb
```

Description:

This notebook:

* Defines the model architectures
* Loads both checkpoints:
* PerformsConfidence based selection
* Best accuracy: 87.03%



## How to Run

### Step 1: Train text model

Open:

```text
train_text_model.ipynb
```

Install all dependency listed in cell 1.

Run all cells to generate:

```text
best_text_model.pth
```

### Step 2: Train Image model

Login to [TALC](https://rcs.ucalgary.ca/TALC_Cluster_Guide)

Install all dependency in a conda env in TALC.  
Modify the slurm file to point to your env and directroy. 
```text
./slurm
```
Use the slurm file to run the script
```text
./best_Image_classification_model.py
```
The image model will be saved in TALC
```text
best_image_model.pth
```

### Step 3: View the result
make sure to have teh data set and model in the correct directory.  
run the following notebook
```text
./confidence_based_selection.ipynb
```

