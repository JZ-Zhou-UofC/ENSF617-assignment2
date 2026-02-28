
---
# Assignment 2 Group 5
# Multi-modal Garbage Classification Project (Image + Text)

## Overview

This project implements a multi-modal classification system that combines both image and text features to improve classification performance. The image model and text model were trained separately, and their outputs are combined in a final notebook to produce the final predictions.

The workflow consists of three main components in addition to the slurm script:

* Slurm batch script
* Text model training (python file trained on TALC)
* Image model training (python file trained on TALC)
* Confidence based selection combining both models (Jupyter Notebook File executed on Google Colab)

---

## Components

Slurm file to run the script on TALC:

Location:

```text
./garbage_class.slurm
```

Slurm script Description:

* Requests a GPU-enabled compute node

* Installs a fresh Conda environment

* Installs PyTorch + ML libraries

* Runs two deep learning Python scripts (image and text)

* Cleans up afterward


### 1. Text Model (Trained on TALC)

Location:

```text
./text_model.py
```

Text Model Description:

* The tutorial text model training is slightly modified to produce a better result
* Transfer learning is used with a pretrained CNN architecture.
* DistilBertModel is used as a base model.
* Checkpoints are saved during training.
* Best accuracy: 85.31%


### 2. Image Model (Trained on TALC)

Location:

```text
./best_Image_classification_model.py
```

Image Model Description:

* The image model is trained on the TALC high-performance computing cluster.
* Images are loaded using PyTorch ImageFolder.
* Transfer learning is used with a pretrained CNN architecture.
* EfficientNet.b2 is choosen to be the based model
* Checkpoints are saved during training.
* Best accuracy: 78.5%


### 3. Confidence based selection (Google Colab)

Location:

```text
./confidence_based_selection_on_assignment1_data.ipynb
```

Confidence based selection Description:

This notebook:

* Defines the model architectures
* Loads both checkpoints:
* PerformsConfidence based selection
* Best accuracy: 87.03%



## How to Run

### Execute the garbage_class.slurm file on TALC.
Modify the slurm file to point to your env and directory.



### Step 1: Train Image model (The slurm file executes best_Image_classification_model.py)

The image model will be saved in TALC
```text
best_image_model.pth
```

### Step 2: Train text model (The slurm file executes text_model.py)

The text model will be saved in TALC
```text
best_text_model.pth
```


### Execute the confidence_based_selection_on_assignment1_data.ipynb on Google Colab
Requirements: <br>
* Load the best_image_model.pth on google drive
* Load the best_text_model.pth on google drive
* Load the garbage dataset on google drive


Instead of averaging predictions, the notebook:

Computes softmax probabilities for:

* Image model

* Text model

* Selects the prediction from the model with higher confidence

* Stores final fused predictions (confidence-based selection (fusion)).

* Computes evaluation such as confusion matrix and also reveals misclassified results.

Also Slurm output file displays logs during image and text training on TALC


