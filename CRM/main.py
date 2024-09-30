import os
import sys
import json
import shutil
from functools import partial
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

# Add the path to MVDream 
sys.path.append('../MVDream')
# sys.path.append('../CRM')  

import random
import numpy as np

# Import necessary modules from MVDream and CRM
from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model

from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from inference import generate3d
from pipelines import TwoStagePipeline
from model import CRM
import rembg

app = FastAPI()

# Mount the 'outputs' directory to serve static files
app.mount("/models", StaticFiles(directory="outputs"), name="models")

# Define the request model
class PromptRequest(BaseModel):
    prompt: str
    uncond_prompt: Optional[str] = ""  # Negative prompt
    seed: Optional[int] = 23
    guidance_scale: Optional[float] = 7.5
    step: Optional[int] = 25
    elevation: Optional[float] = 15
    azimuth: Optional[float] = 0
    use_camera: Optional[bool] = True
    background_choice: Optional[str] = "Auto Remove background"
    foreground_ratio: Optional[float] = 1.0
    backgroud_color: Optional[Tuple[int, int, int]] = (127, 127, 127)

# Global variables to hold the models and configurations
args = None
model = None
sampler = None
pipeline = None
rembg_session = rembg.new_session()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0.0, dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if not isinstance(prompt, list):
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))

def generate_images(args, model, sampler, text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera):
    dtype = torch.float16 if args['fp16'] else torch.float32
    device = args['device']
    batch_size = args['num_frames']

    if use_camera:
        camera = get_camera(args['num_frames'], elevation=elevation, azimuth_start=azimuth)
        camera = camera.repeat(batch_size // args['num_frames'], 1).to(device)
        num_frames = args['num_frames']
    else:
        camera = None
        num_frames = 1

    t = text_input + args['suffix']
    uc = model.get_learned_conditioning([uncond_text_input]).to(device)
    set_seed(seed)

    imgs_list = t2i(
        model,
        args['size'],
        t,
        uc,
        sampler,
        step=step,
        scale=guidance_scale,
        batch_size=batch_size,
        ddim_eta=0.0,
        dtype=dtype,
        device=device,
        camera=camera,
        num_frames=num_frames
    )
    return imgs_list[0]  # Return only the first image

##################### CRM PART ###############################################
def expand_to_square(image: Image.Image, bg_color=(0, 0, 0, 0)) -> Image.Image:
    # Expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def remove_background(image: Image.Image, rembg_session=None, force: bool = False, **rembg_kwargs) -> Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # Explain why currently do not remove background
        print("Alpha channel not empty, skipping background removal, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image.Image, scale_rate: float) -> Image.Image:
    # Resize image content while retaining the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = (
            (original_image.width - resized_image.width) // 2,
            (original_image.height - resized_image.height) // 2
        )
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    # Given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)

def preprocess_image(image: Image.Image, background_choice: str, foreground_ratio: float, backgroud_color: Tuple[int, int, int]) -> Image.Image:
    """
    Input image is a PIL image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")

def CRM_own(inputdir: str, outdir: str, scale: float = 5.0, step: int = 50, bg_choice: str = "Auto Remove background") -> Tuple[str, str]:
    # bg_choice: "Auto Remove background" or "Alpha as mask"
    img = Image.open(inputdir)
    img = preprocess_image(img, bg_choice, 1.0, (127, 127, 127))
    os.makedirs(outdir, exist_ok=True)
    preprocessed_image_path = os.path.join(outdir, "preprocessed_image.png")
    img.save(preprocessed_image_path)

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("configs/specs_objaverse_total.json"))
    # specs = json.load(open("configs/23D.json"))  # Uncomment if using a different config
    crm_model = CRM(specs).to("cuda")
    crm_model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)

    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    # stage1_config = OmegaConf.load("configs/i2is.yaml").config  # Uncomment if using a different config
    # stage2_config = OmegaConf.load("configs/is2ccm.yaml").config  # Uncomment if using a different config

    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    # Download diffusion models
    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth", local_dir='./models/')
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", local_dir='./models/')

    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )

    # Generate stage1 and stage2 images
    rt_dict = pipeline(img, scale=scale, step=step)

    # Stage1 Images
    stage1_images = rt_dict["stage1_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    stage1_image_path = os.path.join(outdir, "pixel_images.png")
    Image.fromarray(np_imgs).save(stage1_image_path)

    # Stage2 Images
    stage2_images = rt_dict["stage2_images"]
    np_xyzs = np.concatenate(stage2_images, 1)
    stage2_image_path = os.path.join(outdir, "xyz_images.png")
    Image.fromarray(np_xyzs).save(stage2_image_path)

    # Generate 3D asset
    glb_path, obj_path = generate3d(crm_model, np_imgs, np_xyzs, "cuda")
    output3d_zip_path = os.path.join(outdir, "output3d.zip")
    shutil.copy(obj_path, output3d_zip_path)

    return glb_path, output3d_zip_path
##############################################################################

@app.on_event("startup")
def startup_event():
    global args, model, sampler

    # Instead of using argparse, define your configuration here
    args = {
        "model_name": "sd-v2.1-base-4view",
        "config_path": None, 
        "ckpt_path": None,    
        "suffix": ", 3d asset",
        "num_frames": 4,
        "size": 256,
        "fp16": False,
        "device": 'cuda'
    }

    print("Loading t2i model...")
    if args["config_path"] is None:
        model = build_model(args["model_name"], ckpt_path=args["ckpt_path"])
    else:
        assert args["ckpt_path"] is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args["config_path"])
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args["ckpt_path"], map_location='cpu'))
    model.device = args["device"]
    model.to(args["device"])
    model.eval()

    sampler = DDIMSampler(model)
    print("t2i model loaded successfully.")

@app.get("/")
async def hello():
    return {"message": "Hello World! This is the MVDream and CRM FastAPI."}

@app.post("/generate3dmodel")
async def generate_image_and_convert_to_3d(request: PromptRequest):
    try:
        # Extract parameters from the request
        prompt = request.prompt
        uncond_prompt = request.uncond_prompt
        seed = request.seed
        guidance_scale = request.guidance_scale
        step = request.step
        elevation = request.elevation
        azimuth = request.azimuth
        use_camera = request.use_camera
        background_choice = request.background_choice
        foreground_ratio = request.foreground_ratio
        backgroud_color = request.backgroud_color

        image = generate_images(
            args,
            model,
            sampler,
            text_input=prompt,
            uncond_text_input=uncond_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            step=step,
            elevation=elevation,
            azimuth=azimuth,
            use_camera=use_camera
        )

        # Save the generated image
        output_dir = "outputs/txt2img-samples/samples/"
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "output.png")
        Image.fromarray(image).save(image_path)

        # Generate 3D model using CRM
        model_3d_glb_path, model_3d_zip_path = CRM_own(
            inputdir=image_path,
            outdir=output_dir,
            scale=5,
            step=50,
            bg_choice=background_choice
        )

        # Construct the download URL
        filename = os.path.basename(model_3d_zip_path)
        download_url = f"http://10.0.154.141:8000/models/txt2img-samples/samples/{filename}"

        return {
            "image_path": image_path,
            "model_3d_glb_path": model_3d_glb_path,
            "model_3d_zip_path": model_3d_zip_path,
            "download_url": download_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
