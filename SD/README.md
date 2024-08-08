# Stable Diffusion Version 2

This repository contains [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [diffusers](https://github.com/huggingface/diffusers).


## Text-to-Image
We remind that before using Stable diffusion you need to download stable diffusion weights. To do so please use [this link](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt) and place the freshly downloaded weights in the checkpoints rtepository under the name "v2-1_768-ema-pruned.ckpt". Otherwise you would have to rename the path in the code...


To generate an image, run the following:

```
python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt ./checkpoints/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768  
```

## Finetuning Stable diffusion using diffusers

If you want to finetune Stable diffusion please follow these instructions

### Building environment

First clone the previous environment you created by following our main README.md
```
conda create --name <new-env> --clone <old-env>
conda activate <new-env>
```
then do 
```
cd diffusers
pip install .
cd example/text_to_image
pip install -r requirements.txt
```
 You'll then need to initilize your accelerate Config with 
``` 
accelerate config
```
and connect to your huggingface account with a 
```
huggingface-cli login
```
Finally you can train your stable diffusion model with this line :

```
accelerate launch --mixed_precision="no"  train_text_to_image_lora.py   --pretrained_model_name="stabilityai/stable-diffusion-2-1"  --resolution=512 --random_flip   --train_batch_size=1   --num_train_epochs=100 --checkpointing_steps=5000 --seed=42 --learning_rate=1e-04   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd_trained_3D_lora" –push_to_hub –hg_path your/hugging_face_path
```



## Acknowledgements
We remind you that almost all the code from these repositories are git clones from [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [diffusers](https://github.com/huggingface/diffusers). Please check their github pages to have more details on how the model works. 