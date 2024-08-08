import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import json
import shutil
import PIL
import rembg
import os
import xformers
import xformers.ops
sys.path.append('../CRM')
from diffusers import StableDiffusionPipeline 
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)

# Then add the lora weights to the model stable diffusion 2
pipe.unet.load_attn_procs('ACROSS-Lab/PromptTo3D_sd_finetuned')
pipe.to("cuda")

##################### CRM PART ###############################################
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from inference import generate3d
from pipelines import TwoStagePipeline
from model import CRM
rembg_session = rembg.new_session()


def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def remove_background(
    image: PIL.Image.Image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def preprocess_image(image, background_choice, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force_remove=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")

def CRM_own(inputdir,  scale = 5.0, step = 50, bg_choice = "Auto Remove background", outdir = "../CRM/out/" ):
   # bg_choice : "[Auto Remove background] or [Alpha as mask]",

    img = Image.open(inputdir)
    img = preprocess_image(img, bg_choice, 1.0, (127, 127, 127))
    os.makedirs(outdir, exist_ok=True)
    img.save(outdir+"preprocessed_image.png")

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("../CRM/configs/specs_objaverse_total.json"))
    #specs = json.load(open("configs/23D.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)

    stage1_config = OmegaConf.load("../CRM/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("../CRM/configs/stage2-v2-snr.yaml").config
    #stage1_config = OmegaConf.load("configs/i2is.yaml").config
    #stage2_config = OmegaConf.load("configs/is2ccm.yaml").config
    
    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    ##potentiellement il y a une nécessité de faire ça en cache en supprimant la partie local_dir ...
    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth", local_dir = '../CRM/models/')
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", local_dir = '../CRM/models/')

    
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )

    
    #dictionnaire en sortie du model avec stage1_image et tout
    rt_dict = pipeline(img, scale=scale, step=step)
    
    ###MV Images ###########################
    stage1_images = rt_dict["stage1_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    Image.fromarray(np_imgs).save(outdir+"pixel_images.png")
    
    ####### CCM Images #################
    stage2_images = rt_dict["stage2_images"]
    np_xyzs = np.concatenate(stage2_images, 1)
    Image.fromarray(np_xyzs).save(outdir+"xyz_images.png")

    ### Génération de l'asset 3D ####################
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")
    shutil.copy(obj_path, outdir+"output3d.zip")
    return glb_path
##############################################################################



#################### Gradio Part #############################################
import gradio as gr


def prompt_to_3D(prompt):
    if prompt is None:
        raise gr.Error("Veuillez rentrer un prompt svp")
    prompt = prompt
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    out_path = "outputs/txt2img-samples/sdf.png"
    image.save(out_path)
    glb_path = CRM_own(out_path)
    return image, glb_path

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder = "Entrez votre pompt ici (en anglais ;) ).")
            generate_image_btn = gr.Button(value="asset 3D")
        with gr.Column():
            image_generee = gr.Image(label = "image 2D", image_mode = 'RGBA', sources = 'upload', type = 'pil', interactive = False)
    with gr.Row():
        output_obj = gr.Model3D(interactive = False, label = "Output 3D asset")
    generate_image_btn.click(fn = prompt_to_3D, inputs = [prompt], outputs = [image_generee, output_obj])
    examples = gr.Examples(examples=["a horse", "a hamburger", "a rabbit", "a man with a blue jacket"], inputs=[prompt])

demo.launch(share=True)

##############################################################################


