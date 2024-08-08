# Prompt to 3D using CRM

## Usage
If you want to use just the CRM model you can run the following command line
```
CUDA_VISIBLE_DEVICES="0" python run.py --inputdir "path/to/your/image.png"
```
otherwise, 3 gradio interfaces using Stable Diffusion, Stable Diffusion Finetuned and MVDream are available runing the following lines
```
python SD_CRM.py
python SDF_CRM.py
python MVD_CRM.py
```

## Finetuning the CRM
 There is a part that is available that allows to finetune the CRM, but we haven't done this yet... If you're interested in finetuning the CRM on data, probably from Objaverse, please check [this repository](https://github.com/SanketDhuri/crm_3d_training). 

## Acknowledgement
 [CRM](https://github.com/thu-ml/CRM) [CRM-finetuned](https://github.com/SanketDhuri/crm_3d_training)
 [MVDream](https://github.com/bytedance/MVDream) [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
