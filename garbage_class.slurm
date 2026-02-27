#!/bin/bash
# ====================================
#SBATCH --job-name=garbage-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-11:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ====================================

# virtual environment

# install conda

# Check if 'software' folder exists and delete it if it does
if [ -d "$HOME/software" ]; then
    echo "The 'software' folder exists. Deleting it now..."
    rm -rf "$HOME/software"
else
    echo "The 'software' folder does not exist."
fi


cd 
mkdir -p software/src
cd software/src

wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-x86_64.sh
bash Miniforge3-24.3.0-0-Linux-x86_64.sh -u -b -p $HOME/software/miniforge3


# initialize conda
eval "$(~/software/miniforge3/bin/conda shell.bash hook)"


# install pytorch
conda create -n pytorch

# activate pytorch
conda activate pytorch

#override auto-detection and force installation of the GPU version.]
CONDA_OVERRIDE_CUDA="12.2" conda install -c conda-forge pytorch=2.4 torchvision python pip

# Install additional libraries
conda install -c conda-forge matplotlib seaborn scikit-learn pillow numpy

# Install the transformers library
conda install -c huggingface transformers

# Install jupyter
conda install jupyter

# Run Python code (replace with your specific Python command and your desired directory)
cd
cd /work/TALC/ensf617_2026w/Lawrencium_X
python best_Image_classification_model.py
python text_model.py


# Deactivate the virtual environment when done
conda deactivate
