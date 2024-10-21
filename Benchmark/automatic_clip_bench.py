import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from glob import glob
from os import path
import trimesh
from tqdm import tqdm
import open_clip
import argparse

# Define device globally for easier access across all functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Similarity:
    def __init__(self, clip_model, tokenizer, preprocess):
        self.clip_model = clip_model.to(device)  # Move model to device at initialization
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def preprocess_image(self, image):
        # Process image and immediately send to device
        return self.preprocess(image).unsqueeze(0).to(device)

    def tokenize_text(self, texts):
        # Tokenize text and immediately send to device
        return self.tokenizer(texts).to(device)

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(preprocessed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        tokenized_texts = self.tokenize_text(texts)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_similarity(self, img_feat, text_feats):
        similarities = torch.cosine_similarity(img_feat, text_feats, dim=-1)
        return similarities 

def process_folder(folder, similarity_model, prompt):
    objects = glob(path.join(folder, '*.obj')) + glob(path.join(folder, '*.glb'))
    results = []
    for obj in objects:
        mesh = trimesh.load(obj, force='mesh')
        screenshots = screenshot_the_mesh(mesh)
        scores = []
        for img in screenshots:
            img_pil = Image.fromarray(img)
            img_feat = similarity_model.encode_image(img_pil)
            text_feat = similarity_model.encode_text([prompt])
            score = similarity_model.compute_similarity(img_feat, text_feat).item()
            scores.append(score)
        top_scores = sorted(scores, reverse=True)[:3]
        avg_score = np.mean(top_scores)
        pipeline, _ = os.path.splitext(path.basename(obj))
        results.append({
            'Folder': path.basename(folder),
            'pipeline': pipeline,
            'Top Score 1': f"{top_scores[0]:.2f}",  
            'Top Score 2': f"{top_scores[1]:.2f}", 
            'Top Score 3': f"{top_scores[2]:.2f}",  
            'Average Score': f"{avg_score:.2f}"    
        })
    return results

def screenshot_the_mesh(mesh):
    resolution = (1024, 768)
    scene = mesh.scene()
    viewpoints = icosahedron_viewpoints()
    screenshots = []
    for viewpoint in viewpoints:
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        try:
            png = scene.save_image(resolution=resolution, visible=True)
            img_arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            screenshots.append(rgb_img)
        finally:
            scene.graph[scene.camera.name] = camera_old
    return screenshots

def icosahedron_viewpoints():
    phi = (1 + np.sqrt(5)) / 2
    vertices = [
        [phi, 1, 0], [phi, -1, 0], [-phi, -1, 0], [-phi, 1, 0],
        [1, 0, phi], [-1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1]
    ]
    viewpoints = []
    for vertex in vertices:
        vertex = np.array(vertex) / np.linalg.norm(vertex)
        angle = np.arccos(np.dot([0, 0, 1], vertex))
        axis = np.cross([0, 0, 1], vertex)
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else [0, 1, 0]
        matrix = trimesh.transformations.rotation_matrix(angle, axis)
        viewpoints.append(matrix)
    return viewpoints

def main(data_folder, prompts_file):
    prompts = []
    with open(prompts_file, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]
    # load the clip model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    clip_model = clip_model.to(device)  # move to the gpu
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    similarity_model = Similarity(clip_model, tokenizer, preprocess) 

    results = []
    folders = sorted(glob(path.join(data_folder, '[0-9]*')))

    # Error handling between incompatible number of subfolders and number of prompts
    
    if len(folders)==len(prompts):
            print("everything is ok")
    elif len(folders)<len(prompts):
        print("Error: Number of folders must match the number of prompts.")
        exit()
    elif len(folders)>len(prompts):
        print("Error: Number of prompts must match the number of folders.")
        exit()
    
    # Processing subfolders and prompts
    for folder, prompt in tqdm(zip(folders, prompts), total=len(folders)):
        
        print("processing folder", folder, "with prompt", prompt)
        folder_results = process_folder(folder, similarity_model, prompt)
        results.extend(folder_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv('Evaluation_clip.csv', index=False)
    print("CSV file has been created with the top scores.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process 3D objects and compute CLIP scores.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the folder containing 3D object files')
    parser.add_argument('--prompts_file', type=str, required=True, help='Path to the file containing prompts')
    
    args = parser.parse_args()
    
    main(data_folder=args.data_folder, prompts_file=args.prompts_file)
