# Prompt to 3D using TRIPOSR [here](https://github.com/VAST-AI-Research/TripoSR) WITH MODIFICATION FOR USING scikit instead of torchmcubes AND stable diffusion [here](https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file)
## Getting Started
### Installation

- Move to SD folder : 
```commandline
cd SD
```

And Then :



```
conda env create -f environment.yaml
conda activate ldm
```
You can update the created environement with 

```
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 
#### xformers efficient attention
For more efficiency and speed on GPUs, 
we highly recommended installing the [xformers](https://github.com/facebookresearch/xformers)
library.

Tested with CUDA 11.4.
Installation needs a somewhat recent version of nvcc and gcc/g++, obtain those, e.g., via 
```commandline
export CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64==9.5.0
```



First, download the weights for [_SD2.1-v_](https://huggingface.co/stabilityai/stable-diffusion-2-1) and [_SD2.1-base_](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and put them into the `PromptTo3D/SD/checkpoints` folder.


### OTHER INSTALATIONS
- Moov to TRIPOSR_COPY

```commandline
cd ../triposr_copy
```


and 

- Install other dependencies by `pip install -r requirements.txt`

Then, run the following. It will take up to 30 min reinsttalling some dependencies 
```commandline
pip install xformers==0.0.16
```
Upon successful installation, the code will automatically default to [memory efficient attention](https://github.com/facebookresearch/xformers)
for the self- and cross-attention layers in the U-Net and autoencoder.

### Manual Inference 
```sh
python pipline.py
```
This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.



### Local Gradio App
```sh
python test_gradio_triposr.py
```

## Troubleshooting

- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.



### For Quick 3D asset generation with MVDream and TripoSR

- Move to MVDream folder :
```commandline
cd ../MVDream
```

And then 
```
pip install -e . --no-deps
```

-Now change Directory 
```commandline
cd ../triposr_copy
```
### Local Gradio App with MVDream

```sh
python MVDream_text_3D_gradio_app.py
```


### Acknowledgement

 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) [TRIPOSR](https://github.com/VAST-AI-Research/TripoSR)





