################################################### THE IMPORTATIONS #############################################################

import os
import torch
from diffusers import StableDiffusionPipeline
import random
import argparse
from functools import partial
import numpy as np
import gradio as gr
import os
from omegaconf import OmegaConf
import torch 
from PIL import Image
import rembg
import torch
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

################################################### THE STABLE DIFFUSION FINE-TUNED #############################################################

def generate_image(prompt, save_dir='generated_images'):
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.unet.load_attn_procs('remi349/sd_trained_3D_lora')
    pipe.to("cuda")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate the image
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    
    # Save the image
    file_path = os.path.join(save_dir, f"image.png")
    image.save(file_path)
    
    return file_path

################################################### THE TripOSR #########################################################################


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


################################################### COMBINATION ################################################################



def generate_and_convert(prompt, output_base='workspace'):
    # Define directories for output
    images_dir = os.path.join(output_base, 'generated_images')
    model_dir = os.path.join(output_base, 'models')
    
    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Generate image
    image_path = generate_image(prompt, save_dir=images_dir)
    print(f"Image generated and saved at: {image_path}")
    
    # Convert image to 3D model
    model_path = process_image(
        input_image_path=image_path,
        output_dir=model_dir,
        remove_bg=True  # Assuming we want to remove the background
    )
    print(f"3D Model generated and saved at: {model_path}")

    return image_path, model_path

# Gradio Interface
def gradio_interface(prompt):
    image_path, model_path = generate_and_convert(prompt)
    return Image.open(image_path), model_path

gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Model3D(label="Download 3D Model")
    ],
    title="Text to Image and 3D Model Generation",
    description="Enter a text prompt to generate an image using Stable Diffusion and convert it into a 3D model."
    
).launch(share=True)
















