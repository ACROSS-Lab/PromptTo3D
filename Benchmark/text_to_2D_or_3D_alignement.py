import os
import numpy as np
import torch
import open_clip
from PIL import Image
import cv2
from glob import glob
from os import path, getcwd
from tqdm import tqdm
import trimesh
import argparse
import platform
import csv
import matplotlib.pyplot as plt

# Setup and configuration for the execution environment
os.environ['DISPLAY'] = ':1'
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CLIP models
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

# Similarity class to compute and process CLIP similarities
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

# Functions to manage 3D model transformations and screenshotting
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


def cube():
    viewpoints = [
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[1, 0, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[1, 0, 0]),
    ]
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
def load_prompts(main_prompts_file, test_prompts_file):
    with open(main_prompts_file, 'r') as file:
        main_prompt = file.readline().strip()  # Lire le premier prompt comme le prompt principal
    with open(test_prompts_file, 'r') as file:
        test_prompts = [line.strip() for line in file.readlines()]  # Lire tous les prompts de test
    return main_prompt, test_prompts


def save_results_to_csv(results, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Index', 'Model Name', 'Prompt', 'Score', 'Rank'])
        for result in results:
            image_index = result['image_index']
            for model_result in result['model_results']:
                for prompt_detail in model_result['prompt_details']:
                    writer.writerow([image_index, model_result['model_name'], prompt_detail['prompt'], prompt_detail['score'], prompt_detail['rank']])

def process_images(images, models, main_prompt, test_prompts, csv_filename):
    results = []
    model_top_prompt_count = {model[3]: 0 for model in models}  # Dictionary to track top prompt counts

    for image_index, image_data in enumerate(tqdm(images)):
        image = Image.open(image_data) if isinstance(image_data, str) else Image.fromarray(image_data).convert("RGB")
        model_results = []

        for model, tokenizer, preprocess, model_name in models:
            processed_image = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(processed_image).squeeze(0)

            prompt_details = []
            scores = []
            for prompt in [main_prompt] + test_prompts:
                tokens = tokenizer(prompt).to(device)
                text_features = model.encode_text(tokens).squeeze(0)
                score = torch.cosine_similarity(image_features, text_features.unsqueeze(0)).item()
                scores.append(score)
                prompt_details.append({'prompt': prompt, 'score': score, 'rank': False})

            # Find index of the highest score
            highest_score_index = np.argmax(scores)
            prompt_details[highest_score_index]['rank'] = True  # Mark the highest score as True

            if highest_score_index == 0:  # If main prompt is ranked top
                model_top_prompt_count[model_name] += 1

            model_results.append({'model_name': model_name, 'prompt_details': prompt_details})

        results.append({'image_index': image_index + 1, 'model_results': model_results})

    save_results_to_csv(results, csv_filename)
    return model_top_prompt_count

def visualize_results(top_prompt_counts, total_images):
    labels = list(top_prompt_counts.keys())
    percentages = [count / total_images * 100 for count in top_prompt_counts.values()]
    plt.bar(labels, percentages, color='blue')
    plt.xlabel('Models')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage of Main Prompt as Top Rank')
    plt.show()


def main(use_screenshots, directory, prompt_path, test_prompt_path, obj_file=None, csv_filename='results.csv'):
    main_prompt, test_prompts = load_prompts(prompt_path, test_prompt_path)
    
    # Define models with their actual names
    models = [
        (clip_model_1, tokenizer_1, preprocess_1, 'ViT-B-32'),   
        (clip_model_2, tokenizer_2, preprocess_2, 'ViT-L-14'),   
        (clip_model_3, tokenizer_3, preprocess_3, 'ViT-B-16'),  
        (clip_model_4, tokenizer_4, preprocess_4, 'ViT-B-16-LAI') 
    ]

    if use_screenshots:
        mesh = trimesh.load(obj_file)
        images = screenshot_the_mesh(mesh) 
    else:
        images = glob(path.join(directory, '*.png'))

    top_prompt_counts = process_images(images, models, main_prompt, test_prompts, csv_filename)
    visualize_results(top_prompt_counts, len(images))

    print("Results:", top_prompt_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP scores for 3D assets.")
    parser.add_argument("--use_screenshots", action="store_true", help="Use screenshots from 3D models instead of directory images")
    parser.add_argument("--directory", type=str, default="./images", help="Directory containing images")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the file containing the main prompt")
    parser.add_argument("--test_prompt_path", type=str, required=True, help="Path to the file containing test prompts")
    parser.add_argument("--obj_file", type=str, help="Path to the OBJ file if using screenshots")
    args = parser.parse_args()

    main(args.use_screenshots, args.directory, args.prompt_path, args.test_prompt_path, args.obj_file)
