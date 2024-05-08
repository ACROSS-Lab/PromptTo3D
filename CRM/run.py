import torch
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from inference import generate3d
from huggingface_hub import hf_hub_download
import json
import argparse
import shutil
from model import CRM
import PIL
import rembg
import os
from pipelines import TwoStagePipeline
import xformers
import xformers.ops

rembg_session = rembg.new_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputdir",
        type=str,
        default="examples/kunkun.webp",
        help="dir for input image",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--bg_choice",
        type=str,
        default="Auto Remove background",
        help="[Auto Remove background] or [Alpha as mask]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out/",
    )
    args = parser.parse_args()

    img = Image.open(args.inputdir)
    img = preprocess_image(img, args.bg_choice, 1.0, (127, 127, 127))
    os.makedirs(args.outdir, exist_ok=True)
    img.save(args.outdir + "preprocessed_image.png")

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)

    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )

    # dictionnaire en sortie du model avec stage1_image et tout
    rt_dict = pipeline(img, scale=args.scale, step=args.step)

    ###MV Images ###########################
    stage1_images = rt_dict["stage1_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    Image.fromarray(np_imgs).save(args.outdir + "pixel_images.png")

    ####### CCM Images #################
    stage2_images = rt_dict["stage2_images"]
    np_xyzs = np.concatenate(stage2_images, 1)
    Image.fromarray(np_xyzs).save(args.outdir + "xyz_images.png")

    ### Génération de l'asset 3D ####################
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")
    shutil.copy(obj_path, args.outdir + "output3d.zip")
