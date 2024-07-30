################################################### THE IMPORTATIONS #############################################################

import sys

# Ajout le chemin du dossier MVDream au système de chemins de Python
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

from PIL import Image
import rembg
import torch
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

################################################### MVDREAM PART #########################################################

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

########################################## TRIPOSR PART ####################################################################
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

    # Renderring if necessary
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

########################################## COMBINATION PART ##########################################################

def generate_image_and_convert_to_3d(prompt, seed, guidance_scale, step, elevation, azimuth, use_camera):
    # Generate Image from prompt via stable diffusion
    image = generate_images(args, model, sampler, prompt, "", seed, guidance_scale, step, elevation, azimuth, use_camera)

    # Save the generated image
    output_dir = "outputs/txt2img-samples/samples"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "output.png")
    Image.fromarray(image).save(image_path)

    # Convert into 3d object with TripoSR model
    output_dir = "output"
    model_3d_path = process_image(image_path, output_dir)

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



######################################################## GRADIO ####################################################################
    with gr.Blocks() as demo:
        gr.Markdown("## MVDream and TripoSR demo for images and 3D generation from text and camera inputs.")
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


