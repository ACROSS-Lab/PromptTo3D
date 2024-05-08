# Prompt to 3D using CRM

This repository contains [Stable Diffusion](https://github.com/CompVis/stable-diffusion) models trained from scratch, [CRM](https://github.com/thu-ml/CRM) and [CRM-finetuned](https://github.com/SanketDhuri/crm_3d_training)

## Requirements

First, you need to have installed conda and to have a CUDA 11.7 installed

Then you should create a conda environment with python version 3.9.19 with a command such as
```
conda create -n <name_environnement> python=3.9
conda activate <name_environnement>
```
After what, make sure you have the 11.4 version of you compiler by running these two command lines 
```
conda install -c conda-forge gxx
conda install gxx=11.4
```
 
 After what please install the right torch versions with these command lines 
 
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
```
And then you can finally install the right xformers version (be carefull to really use the following installation link, it's the only one that worked for me...)
```
pip install git+https://github.com/facebookresearch/xformers@4c06c79#egg=xformers
```
Congrats, you can now clone this project and but it and go inside this repository with a cd. Before installing the requirements, you may need to dowload the stable diffusion particularities with a 
```
pip install -e .
pip install -r requirements.txt
``` 

All in all before doing inference you must download somewhere the weights of stable diffusion model, and place them in the repository, here i downloaded the "v2-1_768-ema-pruned.ckpt" at [this link](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt). 

thus you can try the gradio interface by executing 
```
python test_gradio.py
```

## Finetuning the CRM
 There is a part that is available that allows to finetune the CRM, but we have'nt done this yet...

### Acknowledgement
 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) [CRM](https://github.com/thu-ml/CRM)
 [CRM-finetuned](https://github.com/SanketDhuri/crm_3d_training)
