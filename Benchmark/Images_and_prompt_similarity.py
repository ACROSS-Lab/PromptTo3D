import trimesh
import numpy as np
from PIL import Image
from glob import glob
from os import path, getcwd
from tqdm import tqdm
import argparse
import cv2
import os
import torch
import open_clip
import matplotlib.pyplot as plt

os.environ['DISPLAY'] = ':1'
import platform

if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize open_clip components for four different models
clip_model_1, _, preprocess_1 = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model_1 = clip_model_1.to(device)
clip_model_1.eval()
tokenizer_1 = open_clip.get_tokenizer('ViT-B-32')

clip_model_2, _, preprocess_2 = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model_2 = clip_model_2.to(device)
clip_model_2.eval()
tokenizer_2 = open_clip.get_tokenizer('ViT-L-14')

clip_model_3, _, preprocess_3 = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
clip_model_3 = clip_model_3.to(device)
clip_model_3.eval()
tokenizer_3 = open_clip.get_tokenizer('ViT-B-16')

clip_model_4, _, preprocess_4 = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e31')
clip_model_4 = clip_model_4.to(device)
clip_model_4.eval()
tokenizer_4 = open_clip.get_tokenizer('ViT-B-16')

class Similarity:
    def __init__(self, clip_model, tokenizer, preprocess):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def preprocess_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(device)
        return image

    def tokenize_text(self, texts):
        inputs = self.tokenizer(texts)
        return inputs.to(device)

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

    def forward(self, image, captions):
        img_feat = self.encode_image(image)
        text_feats = self.encode_text(captions)
        similarities = self.compute_similarity(img_feat, text_feats)
        return similarities

def cube():
    viewpoints = [
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[1, 0, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[1, 0, 0]),
    ]
    return viewpoints

def icosahedron():
    phi = (1+np.sqrt(5))/2
    vertices = [[phi, 1, 0], [phi, -1, 0], [-phi, -1, 0], [-phi, 1, 0],
                [1, 0, phi], [-1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
                [0, phi, 1], [0, phi, -1], [0, phi, -1], [0, -phi, 1]]
    vertices = [np.array(e) for e in vertices]
    X = np.array([1,0,0])
    O = np.array([0,0,0])
    viewpoints = []
    for e in vertices :
        XO = O - X
        YO = O - e
        XO = XO / np.linalg.norm(XO)
        YO = YO / np.linalg.norm(YO)
        angle  = np.arccos(np.clip(np.dot(XO, YO),-1,1))
        direction = np.cross(XO,YO)
        viewpoints.append(trimesh.transformations.rotation_matrix(angle=angle, direction=direction))
    return viewpoints

def screenshot_the_mesh(mesh, out_folder='./imagesout_01/', method=icosahedron, save_views=False):
    resolution = (1024, 768)
    filename_format = "{:03d}.png"
    scene = mesh.scene()
    viewpoints = method()
    screenshots = []
    for i, viewpoint in enumerate(viewpoints):
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        filename = filename_format.format(i)
        try:
            png = scene.save_image(resolution=resolution, visible=True)
            img_arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            screenshots.append(rgb_img)
            if save_views:
                with open(path.join(out_folder, filename), "wb") as f:
                    f.write(png)
            scene.graph[scene.camera.name] = camera_old
        except ZeroDivisionError:
            print("Error: Window resizing caused division by zero. Try setting minimum window size or handling resizing events.")
    return screenshots

def count_polygons(mesh):
    return len(mesh.faces)

def count_vertices(mesh):
    return len(mesh.vertices)

def main(method, folder_mother, name, save_views):
    cwd = getcwd()
    folder_mother = path.relpath(folder_mother, cwd)
    
    with open(path.join(folder_mother, 'prompts.txt'), 'r') as f:
        prompts = [line.strip() for line in f.readlines()]  # Read multiple prompts
    
    if len(prompts) < len(glob(path.join(folder_mother, '[0-9]*/'))):
        raise ValueError("Le nombre de prompts doit être égal au nombre d'images capturées.")

    similarity_model_1 = Similarity(clip_model_1, tokenizer_1, preprocess_1)
    similarity_model_2 = Similarity(clip_model_2, tokenizer_2, preprocess_2)
    similarity_model_3 = Similarity(clip_model_3, tokenizer_3, preprocess_3)
    similarity_model_4 = Similarity(clip_model_4, tokenizer_4, preprocess_4)

    all_scores_1 = []
    all_scores_2 = []
    all_scores_3 = []
    all_scores_4 = []

    prompt_scores_1 = []
    prompt_scores_2 = []
    prompt_scores_3 = []
    prompt_scores_4 = []

    polygon_counts = []
    vertex_counts = []

    for folder_idx, folder in enumerate(tqdm(sorted(glob(path.join(folder_mother, '[0-9]*/'))))):
        folder_name = path.relpath(folder, cwd)
        prompt = prompts[folder_idx]
        if name:
            asset_path = path.join(folder_name, name + '.obj')
            if not path.isfile(asset_path):
                raise FileNotFoundError(f"Le dossier '{folder_name}' est incomplet, il devrait contenir : '{name}.obj'")
        else:
            asset_path = glob(path.join(folder, "*.obj"))[0]
        
        mesh = trimesh.load(asset_path)
        polygon_count = count_polygons(mesh)
        vertex_count = count_vertices(mesh)
        polygon_counts.append(polygon_count)
        vertex_counts.append(vertex_count)

        screenshots = screenshot_the_mesh(mesh=mesh, out_folder=folder, method=method, save_views=save_views)
        
        with open(path.join(folder, 'similarity_scores.txt'), 'w') as f:
            f.write(f"Polygon count: {polygon_count}\n")
            f.write(f"Vertex count: {vertex_count}\n")
            for i, screenshot in enumerate(screenshots):
                screenshot_pil = Image.fromarray(screenshot)
                captions = [prompt]
                
                similarity_scores_1 = similarity_model_1.forward(screenshot_pil, captions)
                similarity_scores_2 = similarity_model_2.forward(screenshot_pil, captions)
                similarity_scores_3 = similarity_model_3.forward(screenshot_pil, captions)
                similarity_scores_4 = similarity_model_4.forward(screenshot_pil, captions)

                score_prompt_1 = float(similarity_scores_1[0].cpu().numpy())
                score_prompt_2 = float(similarity_scores_2[0].cpu().numpy())
                score_prompt_3 = float(similarity_scores_3[0].cpu().numpy())
                score_prompt_4 = float(similarity_scores_4[0].cpu().numpy())
                
                prompt_scores_1.append(score_prompt_1)
                prompt_scores_2.append(score_prompt_2)
                prompt_scores_3.append(score_prompt_3)
                prompt_scores_4.append(score_prompt_4)

                f.write(f"Image {i+1} Model 1 (ViT-B-32): CLIP similarity (Prompt): {score_prompt_1}, Prompt: {prompt}\n")
                f.write(f"Image {i+1} Model 2 (ViT-L-14): CLIP similarity (Prompt): {score_prompt_2}, Prompt: {prompt}\n")
                f.write(f"Image {i+1} Model 3 (ViT-B-16): CLIP similarity (Prompt): {score_prompt_3}, Prompt: {prompt}\n")
                f.write(f"Image {i+1} Model 4 (ViT-B-16 laion400m_e31): CLIP similarity (Prompt): {score_prompt_4}, Prompt: {prompt}\n")
                print(f"Image {i+1} Model 1 (ViT-B-32): CLIP similarity (Prompt): {score_prompt_1}, Prompt: {prompt}")  
                print(f"Image {i+1} Model 2 (ViT-L-14): CLIP similarity (Prompt): {score_prompt_2}, Prompt: {prompt}")  
                print(f"Image {i+1} Model 3 (ViT-B-16): CLIP similarity (Prompt): {score_prompt_3}, Prompt: {prompt}")  
                print(f"Image {i+1} Model 4 (ViT-B-16 laion400m_e31): CLIP similarity (Prompt): {score_prompt_4}, Prompt: {prompt}")  

    mean_prompt_score_1 = np.mean(prompt_scores_1)
    mean_prompt_score_2 = np.mean(prompt_scores_2)
    mean_prompt_score_3 = np.mean(prompt_scores_3)
    mean_prompt_score_4 = np.mean(prompt_scores_4)

    print(f"Model 1 (ViT-B-32): Moyenne des scores CLIP basés sur les prompts : {mean_prompt_score_1}")
    print(f"Model 2 (ViT-L-14): Moyenne des scores CLIP basés sur les prompts : {mean_prompt_score_2}")
    print(f"Model 3 (ViT-B-16): Moyenne des scores CLIP basés sur les prompts : {mean_prompt_score_3}")
    print(f"Model 4 (ViT-B-16 laion400m_e31): Moyenne des scores CLIP basés sur les prompts : {mean_prompt_score_4}")
    print(f"Le nombre de polygones est : {(polygon_counts)}")
    print(f"Le nombre de sommets est : {(vertex_counts)}")

    # Visualisation des scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(prompt_scores_1) + 1), prompt_scores_1, marker='o', linestyle='-', label='Model 1 (ViT-B-32)')
    plt.plot(range(1, len(prompt_scores_2) + 1), prompt_scores_2, marker='o', linestyle='--', label='Model 2 (ViT-L-14)')
    plt.plot(range(1, len(prompt_scores_3) + 1), prompt_scores_3, marker='o', linestyle=':', label='Model 3 (ViT-B-16)')
    plt.plot(range(1, len(prompt_scores_4) + 1), prompt_scores_4, marker='o', linestyle='-.', label='Model 4 (ViT-B-16 laion400m_e31)')
    plt.axhline(y=mean_prompt_score_1, color='r', linestyle='--', label='Mean Model 1 (ViT-B-32)')
    plt.axhline(y=mean_prompt_score_2, color='g', linestyle='--', label='Mean Model 2 (ViT-L-14)')
    plt.axhline(y=mean_prompt_score_3, color='m', linestyle='--', label='Mean Model 3 (ViT-B-16)')
    plt.axhline(y=mean_prompt_score_4, color='c', linestyle='--', label='Mean Model 4 (ViT-B-16 laion400m_e31)')
    plt.title('CLIP Similarity Scores for Images')
    plt.xlabel('Image Index')
    plt.ylabel('CLIP Similarity Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.savefig('clip_similarity_scores.png')
    plt.show()

    # Visualisation des nombres de polygones et de sommets
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(polygon_counts) + 1), polygon_counts, marker='o', linestyle='-', color='b', label='Polygon Count')
    plt.plot(range(1, len(vertex_counts) + 1), vertex_counts, marker='x', linestyle='--', color='r', label='Vertex Count')
    plt.title('Polygon and Vertex Counts for 3D Objects')
    plt.xlabel('Object Index')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    plt.savefig('polygon_vertex_counts.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculer les scores CLIP des captures d’écran des actifs 3D')
    parser.add_argument("--method", help="Vues prises, que ce soit celles d'un icosahedron (12 sommets) ou d'un cube (6 faces)...", default=icosahedron)
    parser.add_argument("--folder_mother", help="Dossier dans lequel sont situées les meshes de nos objets 3D", default='./', type=str)
    parser.add_argument("--name", help="Nom des fichiers .obj", default=None, type=str)
    parser.add_argument("--save_views", default=False, type=bool, help="Booléen qui décide si oui ou non on sauvegarde les vues de notre modèle")
    args = parser.parse_args()
    main(**vars(args))
