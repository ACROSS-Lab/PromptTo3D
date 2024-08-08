"""
Script that transforms an asset into a gif
the gif will be saved at out_path

"""
import subprocess
import trimesh
import numpy as np
import requests
from PIL import Image
import contextlib

from glob import glob
from os import path, getcwd
from tqdm import tqdm
import argparse
import cv2
import os
import shutil

def start_xvfb():
    cmd = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &"
    subprocess.Popen(cmd, shell=True) 

def sliced(n_iter = 100):
    """
    Method that allows to takes pictures of the mesh, n_iter is the number of pictures taken,
    at regular intervals following a rotation around the mesh.
    @param n_iter : int, number of views of screenshots to take
    """
    viewpoints = [
        trimesh.transformations.rotation_matrix(angle=2*i*np.pi/n_iter, direction=[0.5, 0.5, 0])
        for i in range (n_iter)

    ]
    return viewpoints

def screenshot_the_mesh(mesh, out_folder, save_views = True):
    """
    Script that takes screenshots from a trimesh mesh and that may save them
    @param mesh : trimesh mesh, mesh that we want to take screenshots from
    @param out_folder : str, where the images may be stored if saved
    @param save_views : Bool, argument that decides if we save the images taken at out_folder
    
    """
    os.environ['DISPLAY'] = ':99'
    os.environ['LC_ALL'] = 'C'
    resolution=(512, 384)
    filename_format="{:03d}.png"
    #the fact to create a Trimesh Scene allows us to load .glb meshes nd .obj meshes
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    viewpoints = sliced()
    # Create the output folder if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)
    screenshots = [] # possible improvement : maybe consider a generator
    for i, viewpoint in enumerate(viewpoints):
        # move the camera
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        # save images if needed
        filename = filename_format.format(i)
        try :
            #image = scene.export_scene(resolution=(width, height))
            #image.save("output.png")
            png = scene.save_image(resolution=resolution, visible=False)
            img_arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            screenshots.append(rgb_img)
            if save_views :
                with open(path.join(out_folder,filename), "wb") as f:
                    f.write(png)
            #going back to initial camera pose
            scene.graph[scene.camera.name] = camera_old
        except ZeroDivisionError:
            print("Error: Window resizing caused division by zero. Try setting minimum window size or handling resizing events.")
    return screenshots

def main(path_mesh, save_views, out_path):
    """
    This main scripts creates many gifs from the .obj file saved at location folder_mother
    @param folder_mother : str, where the images are saved
    @save_views : Bool, save or not the images (here we always save them)
    """
    #need to have downloaded xvfb
    start_xvfb()
    os.environ['DISPLAY'] = ':99'
    os.environ['LC_ALL'] = 'C'
    cwd = getcwd()
    folder_mother = path.relpath(folder_mother, cwd)
    mesh = trimesh.load(asset_path)
    screenshots = screenshot_the_mesh(mesh = mesh, out_folder = "./tmp/", save_views = save_views)
    filenames = [f"./tmp/{i:03d}.png" for i in range (100)]
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f)) for f in filenames)
        img = next(imgs)
        img.save(fp=out_path, format='GIF', append_images=imgs, save_all=True, duration=40, loop=0)
    #then remove all the images created
    shutil.rmtree('./tmp')
    print(f"Your gif has been saved at the location you provided {out_path}")

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description = 'Transform a 3D mesh into a gif')
    parser.add_argument(
        "--path_mesh",
        help = "path of the 3D mesh",
        default = './mesh.obj',
        type = str
    )
    parser.add_argument(
        "--save_views",
        default = True,
        type = bool,
        help = "boolean that decides if we save the screenshots taken, for now the script only works if we download the views"
    )
    parser.add_argument(
        "--out_path",
        default = "./mesh.gif",
        type = str,
        help = "path of the gif to save"
    )
    args = parser.parse_args()
    main(**vars(args))