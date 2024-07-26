import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder


from PIL import Image
from huggingface_hub import hf_hub_download
import json
import shutil
import PIL
import rembg
import os
import xformers
import xformers.ops

import gradio as gr

import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video





##########################STABLE DIFFUSION PART #######################################
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def stable_diffusion_t2i(prompt, outdir = "outputs/txt2img-samples", 
                         steps = 50, config = "../SD/configs/stable-diffusion/v2-inference-v.yaml",
                         ckpt = "../SD/checkpoints/v2-1_768-ema-pruned.ckpt", ddim_eta = 0.0,
                         n_iter = 1, H = 768, W = 768, device = "cuda", 
                         C = 4, f=8, n_samples = 1, n_rows = 0, scale = 9.0,
                         seed = 42, precision = "autocast", repeat = 1, 
                        torchscript = False, ipex = False, bf16 = False, 
                        plm = False, dpm = False, fixed_code = False):
    """
    Algo de Stable diffusion
    @param prompt: str, le prompt d'entrée
    @param outdir: str, le fichier où on enregistre l'image
    @param steps: int, number of ddim sampling steps, à augmenter ???
    @param ckpt: str, heckpoint du model
    @param ddim_eta: float, eta=0.0 corresponds to deterministic sampling
    @param n_iter: (sample this often ??) -> mettre 3 pour avoir plus d'image
    @param H: float, hauteur de l'image générée
    @param W: float, Largeur de l'image
    @param device: str, device (cpu ou gpu)
    @param C: int, nombre de latent chanels
    @param f: int, downsampling factor (8 ou 16)
    @param n_samples: int, batchsize
    @param n_rows: int, lignes dans la grilles d'images
    @param scale: float, unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    @param seed: int, caractère aléatoire, 42 = reproductible
    @param precision: str, evaluate at this precision, full ou autocast
    @param repeat: int, repéter chaque prompt du fichier repeat fois
    @param torchscript: Bool, Use TorchScript, pre optimized code... à tester
    @param ipex: Bool, intel extension for pytorch, plus rapide si on a un ordi intel
    @param bf16: Bool, réduire qualité
    @param plm: Bool, à tester, plm sampling
    @param dpm: Bool, à tester, dpm sampling
    @param fixed_code: Bool,if enabled, uses the same starting code across all samples
    


    """
    
    seed_everything(seed)

    config = OmegaConf.load(f"{config}")
    device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{ckpt}", device)

    if plm:
        sampler = PLMSSampler(model, device=device)
    elif dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    prompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    if torchscript or ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if bf16 else nullcontext()
        shape = [C, H // f, W // f]

        if bf16 and not torchscript and not ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if ipex:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if torchscript:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                    "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if precision=="autocast" or bf16 else nullcontext
   
    device = "cuda"
    with torch.no_grad(), \
        precision_scope(device), \
        model.ema_scope():
            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples, _ = sampler.sample(S=steps,
                                                     conditioning=c,
                                                     batch_size=n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta,
                                                     x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, "output.png"))
                        

                    all_samples.append(x_samples)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    





#####################################################################################
#                                                                                   #
#                       TRIPOSR                                                     #
#                                                                                   #
#####################################################################################





def process_image(input_image_path, output_dir, device='cuda:0', pretrained_model='stabilityai/TripoSR', chunk_size=8192,
                  mc_resolution=256, remove_bg=True, foreground_ratio=0.85, model_save_format='obj', render=False):
    os.makedirs(output_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        device = "cpu"
    
    # Initialisation of TSR model
    model = TSR.from_pretrained(
        pretrained_model,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)

    # Image processing
    if remove_bg:
        rembg_session = rembg.new_session()
    else:
        rembg_session = None

    image = Image.open(input_image_path)
    if remove_bg:
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        input_image_path = os.path.join(output_dir, "input.png")
        image.save(input_image_path)

    # TSR model execution
    with torch.no_grad():
        scene_codes = model([image], device=device)

    # Renderring if neccessary
    if render:
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))
        save_video(render_images[0], os.path.join(output_dir, "render.mp4"), fps=30)

    # Exportation of the Mesh
    meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)
    output_mesh_path = os.path.join(output_dir, f"mesh.{model_save_format}")
    meshes[0].export(output_mesh_path)
    print("Mesh exportation ended")

    return output_mesh_path





###################################################             GRADIO              ############################################################################################



def generate_image_and_convert_to_3d(prompt):
    # Generate Image from prompt via stable diffusion
    image_path = stable_diffusion_t2i(prompt, outdir="outputs/txt2img-samples")
    
    # Verify if the image is correctly generated 
    image_file = os.path.join("outputs/txt2img-samples/samples", "output.png")
    if not os.path.exists(image_file):
        return "Aucune image générée, veuillez réessayer avec un autre prompt.", None
    
    # Convert into 3d object with TripoSR model
    output_dir = "output"
    model_3d_path = process_image(image_file, output_dir)
    
    return image_file, model_3d_path

# Use gr.Blocks for l'interface construction
with gr.Blocks() as demo:
    gr.Markdown("## Generate a 3D Object from a Prompt")
    gr.Markdown("Enter a prompt to generate an image, which will then be converted into a 3D model.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here by describing what exactly you want, e.g., car standing up.")
            generate_image_btn = gr.Button("Generate 3D")
        with gr.Column():
            image_generated = gr.Image(label="Generated Image")
            model_3d = gr.Model3D(label="3D Model")

    def update_outputs(prompt):
        image_path, model_3d_path = generate_image_and_convert_to_3d(prompt)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = None
        if model_3d_path and os.path.exists(model_3d_path):
            model_path = model_3d_path
        else:
            model_path = None
        return image, model_path

    generate_image_btn.click(
        fn=update_outputs,
        inputs=prompt,
        outputs=[image_generated, model_3d]
    )

demo.launch(share=True)
