import cv2
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import torch
from huggingface_hub import hf_hub_download
import json
import shutil
import PIL
import rembg
import xformers
import xformers.ops





import sys

# Ajouter le chemin du dossier MVDream au système de chemins de Python
sys.path.append('../MVDream')
import random
import argparse
from functools import partial
import numpy as np
import gradio as gr
import os
from omegaconf import OmegaConf
import torch 

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model




def set_seed(seed):
    seed = int(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                         batch_size=batch_size, shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc_,
                                         eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def generate_images(args, model, sampler, text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera):
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    if use_camera:
        camera = get_camera(args.num_frames, elevation=elevation, azimuth_start=azimuth)
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
        num_frames = args.num_frames
    else:
        camera = None
        num_frames = 1

    t = text_input + args.suffix
    uc = model.get_learned_conditioning([uncond_text_input]).to(device)
    set_seed(seed)

    imgs_list = t2i(model, args.size, t, uc, sampler, step=step, scale=guidance_scale,
                    batch_size=batch_size, ddim_eta=0.0, dtype=dtype, device=device,
                    camera=camera, num_frames=num_frames)
    return imgs_list[0]  # Return only the first image

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








def CRM_own(inputdir, outdir, scale = 5.0, step = 50, bg_choice = "Auto Remove background" ):

   # bg_choice : "[Auto Remove background] or [Alpha as mask]",

    img = Image.open(inputdir)
    img = preprocess_image(img, bg_choice, 1.0, (127, 127, 127))
    os.makedirs(outdir, exist_ok=True)
    img.save(outdir+"preprocessed_image.png")

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("configs/specs_objaverse_total.json"))
    #specs = json.load(open("configs/23D.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)

    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    #stage1_config = OmegaConf.load("configs/i2is.yaml").config
    #stage2_config = OmegaConf.load("configs/is2ccm.yaml").config

    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    ##potentiellement il y a une nécessité de faire ça en cache en supprimant la partie local_dir ...
    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth", local_dir = './models/')
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", local_dir = './models/')


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



def generate_image_and_convert_to_3d(prompt, seed, guidance_scale, step, elevation, azimuth, use_camera):
    # Generate Image from prompt via stable diffusion
    image = generate_images(args, model, sampler, prompt, "", seed, guidance_scale, step, elevation, azimuth, use_camera)

    # Save the generated image
    image_path = "outputs/txt2img-samples/samples/output.png"
    os.makedirs(image_path[:-11], exist_ok=True)
    Image.fromarray(image).save(image_path)
    model_3d_path = CRM_own(image_path, "./outputs")

    Image.fromarray(image).save(image_path)

    return image_path, model_3d_path, model_3d_path  # Return image path, model path for visualization, and model path for download



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = args.device
    model.to(args.device)
    model.eval()

    sampler = DDIMSampler(model)
    print("load t2i model done . ")

    fn_with_model = partial(generate_images, args, model, sampler)

    with gr.Blocks() as demo:
        gr.Markdown("MVDream and TripoSR demo for images and 3D generation from text and camera inputs.")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(value="", label="Prompt")
                uncond_text_input = gr.Textbox(value="", label="Negative prompt")
                seed = gr.Number(value=23, label="Seed", precision=0)
                guidance_scale = gr.Number(value=7.5, label="Guidance scale")
                step = gr.Number(value=25, label="Sample steps", precision=0)
                elevation = gr.Slider(0, 30, value=15, label="Elevation")
                azimuth = gr.Slider(0, 360, value=0, label="Azimuth")
                use_camera = gr.Checkbox(value=True, label="Multi-view Mode")
                generate_button = gr.Button("Generate Images and Convert to 3D")

            image_output = gr.Image(label="Generated Image")  # Create a single image component for generated image
            model_output = gr.Model3D(label="3D Model")  # Create a file visualization component for the 3D model
            download_output = gr.File(label="Download OBJ")  # Create a file component for downloading the 3D model



        inputs = [text_input, seed, guidance_scale, step, elevation, azimuth, use_camera]
        outputs = [image_output, model_output, download_output]
        generate_button.click(generate_image_and_convert_to_3d, inputs=inputs, outputs=outputs)

    demo.launch(share=True)