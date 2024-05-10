# Prompt to 3D using TRIPOSR [here](https://github.com/VAST-AI-Research/TripoSR) WITH MODIFICATION FOR USING scikit instead of torchmcubes AND 
stable diffusion [here](https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file)
## Getting Started
### Installation
- Move to SD folder : cd SD
'''
conda env create -f environment.yaml
conda activate ldm
'''
You can update the created environement with 
```
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch
pip install transformers==4.28.1 diffusers invisible-watermark
pip install -e .
``` 
#### xformers efficient attention
For more efficiency and speed on GPUs, 
we highly recommended installing the [xformers](https://github.com/facebookresearch/xformers)
library.

Tested with CUDA 11.7.
Installation needs a somewhat recent version of nvcc and gcc/g++, obtain those, e.g., via 
```commandline
export CUDA_HOME=/usr/local/cuda-11.7
conda install -c nvidia/label/cuda-11.7 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64==9.5.0
```

Then, run the following.
'''
pip install xformers==0.0.16
'''
Upon successful installation, the code will automatically default to [memory efficient attention](https://github.com/facebookresearch/xformers)
for the self- and cross-attention layers in the U-Net and autoencoder.

### OTHER INSTALATIONS
- Moov to TRIPOSR_COPY
'''
cd ../triposr_copy
'''
and do 
- Install other dependencies by `pip install -r requirements.txt`

### Manual Inference 
```sh
python pipline.py
```
This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.



### Local Gradio App
```sh
python test_gradio_app.py
```

## Troubleshooting

- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.




### Acknowledgement

 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) [TRIPOSR](https://github.com/VAST-AI-Research/TripoSR)

