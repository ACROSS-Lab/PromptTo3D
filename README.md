# PromptTo3D
In the context of the Simple project, it is needed to have an efficient model that allows to generate a 3D asset from a prompt.
Thus we adapted two models; the CRM and the triposr, that are image to 3D assets models. We use a stablediffusion, stable diffusion finetuned and MVDream models for the prompt to image part. In order to create a 3D asset from a prompt, please check the README.md files in the CRM and the LRM folders. We are now going to build an environment that works with both CRM and triposr models. 


## Requirements

First, you need to have installed conda and to have a CUDA 11.7 installed
To make sure of it the command line 
```
nvcc --version
```
is supposed to return something like 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```
otherwise, if you don't have this version installed, please install it following the instructions on [this page](https://developer.nvidia.com/cuda-11-7-0-download-archive), and make sure to have installed a cuda toolkit too, that the PATH is well defined to the right cuda version and that the variable CUDA_HOME points to the right cuda version too (⚠⚠⚠ make sure this path is correct and exists). Try once again the nvcc --version command.

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
Please, make sure that your torch is well linked to your cuda version, by running this on your terminal :
```
python -c "import torch; print(torch.cuda.is_available())"
```
it is supposed to return true, otherwise there is an issue that needs to be fixed to continue.


Congrats, you can now clone this project if that's not already done
```
git clone https://github.com/ACROSS-Lab/PromptTo3D.git
cd PromptTo3D
```
Please install the requirements along with NVDiffrast library and xformers library (be carefull to really use the following installation link, it's the only one that works with this environment...)
```
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast
pip install git+https://github.com/facebookresearch/xformers@4c06c79#egg=xformers
``` 
Please, keep in mind that this only works if you have a great wifi connection. Otherwise you will have an error like (that happens in the IRD labo in Thuy Loi...):
```
error: XXXX bytes of body are still expected
```
If this error happens you just need to rerun the previous command when you have a better connection, xformers is very capricious...

All in all before doing inference you must download somewhere the weights of stable diffusion model, and place them in the repository, here i downloaded the "v2-1_768-ema-pruned.ckpt" at [this link](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt). 
IMPORTANT : these weights must be placed in the checkpoints folder of the SD folder, and named this very same way as otherwise you would need to change the path of model inside the scripts

You also need to install stable diffusion and MVDream by going to the right folders 
```
cd SD
pip install -e .
cd ../MVDream
pip install -e . --no-deps
```
Congrats, now you can go in each of the CRM and Triposr folder to execute the right script to generate your assets.

For the part about GIF generation from 3D assets, we decided to use a virtual framebuffer, in order for the code to be able to run on displayless devices, such as the server we used... Thus you need to install xvfb fo rthat part, you can do so by running the following : 

```
sudo apt update
sudo apt install xvfb
```


Now you can try our gradio interfaces located in the file scripts by doing for example :
```
cd scripts
python SD_CRM.py
```

### Acknowledgement
 [CRM](https://github.com/thu-ml/CRM) [CRM-finetuned](https://github.com/SanketDhuri/crm_3d_training) 
 [Diffusers](https://github.com/huggingface/diffusers) [MVDream](https://github.com/bytedance/MVDream)
 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) [T3Bench] (https://github.com/THU-LYJ-Lab/T3Bench)
 [Triposr] (https://github.com/VAST-AI-Research/TripoSR) 
