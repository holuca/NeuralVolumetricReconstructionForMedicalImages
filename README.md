# Laminography-Adapted NAF
This is a framework for reconstructing attenutation field using NAF. It has been adapted to work for laminography settings.

## Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n naf python=3.9 -y
conda activate naf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```

## Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n naf python=3.9 -y
conda activate naf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```

## Training and evaluation

Download four CT datasets from [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing). Put them into the `./data` or `./data_npy` folder respecively.

Experiments settings are stored in `./config` folder.

For example, train NAF with `brain` dataset:

``` sh
python train.py --config ./config/brain.yaml
```

The evaluation outputs will be saved in `./logs/eval/epoch_*` folder.

In the yaml file, change the last activation function to tanh, none respectively and data needs to be normalized if needed. Don't forget that the brain dataset is only using phase for its reconstruction when no masks used.  

## Coordinate system
Our coordinate system is similar to that in TIGRE toolbox, except for the detector plane which follows OpenCV standards.

![NAF coordinate systen](assets/coord.png)


## Acknowledgement

* Hash encoder and code structure are adapted from [torch-ngp](https://github.com/ashawkey/torch-ngp.git).
* Many thanks to the amazing [TIGRE toolbox](https://github.com/CERN/TIGRE.git).). Put them into the `./data` folder.
* All code has been adapted from [NAF](https://github.com/Ruyi-Zha/naf_cbct).
  
