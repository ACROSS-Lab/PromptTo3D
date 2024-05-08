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
    #### SOLUTIOn SEULEMENT TEMPORAIRE
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
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                        sample_count += 1

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    
    

##############################################################################


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

## implémenter la possiblilité d'avoir plusieurs images !!!!







def CRM_own(inputdir,  scale = 5.0, step = 50, bg_choice = "Auto Remove background", outdir = "out/" ):

   # bg_choice : "[Auto Remove background] or [Alpha as mask]",

    img = Image.open(inputdir)
    img = preprocess_image(img, bg_choice, 1.0, (127, 127, 127))
    os.makedirs(outdir, exist_ok=True)
    img.save(outdir+"preprocessed_image.png")

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)

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
#PRE_PROMPT = ""
#POST_PROMPT = ""

PRE_PROMPT = "i want to create a 3D asset from this prompt by first generating an image, create a "
POST_PROMPT = "standing from far and isolated with lighting everywhere no sun"
#PRE_PROMPT = "I want to create a 3D asset from this prompt by first generating an image, create"
#POST_PROMPT = "full, whole and complete, standing from very far and isolated with lighting everywhere, and a solid background please"
def prompt_to_image(prompt):
    if prompt is None:
        raise gr.Error("Veuillez rentrer un prompt svp")
    prompt = PRE_PROMPT + prompt + POST_PROMPT
    stable_diffusion_t2i(prompt = prompt)
    #do resize content ? expand to square ?
    out_path = "outputs/txt2img-samples/samples"
    grid_count = len(os.listdir(out_path)) - 1
    out_path = out_path + f"/{grid_count:05}.png"
    img = Image.open(out_path)
    return img

def prompt_to_image2(prompt):
    if prompt is None:
        raise gr.Error("Veuillez rentrer un prompt svp")
    prompt = prompt + POST_PROMPT
    stable_diffusion_t2i(prompt = prompt)
    #do resize content ? expand to square ?
    out_path = "outputs/txt2img-samples/samples"
    grid_count = len(os.listdir(out_path)) - 1
    out_path = out_path + f"/{grid_count:05}.png"
    img = Image.open(out_path)

    glb_path = CRM_own(out_path)


    return img, glb_path


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder = "Entrez votre pompt ici (en anglais ;) ).")
            generate_image_btn = gr.Button(value="asset 3D")


        with gr.Column():
            image_generee = gr.Image(label = "image 2D", image_mode = 'RGBA', sources = 'upload', type = 'pil', interactive = False)
    with gr.Row():
        output_obj = gr.Model3D(interactive = False, label = "Output 3D asset")
    generate_image_btn.click(fn = prompt_to_image2, inputs = [prompt], outputs = [image_generee, output_obj])
    examples = gr.Examples(examples=["a horse", "a hamburger", "a rabbit", "a man with a blue jacket"], inputs=[prompt])



demo.launch(share=True)

##############################################################################


