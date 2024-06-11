import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import trimesh
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import argparse
from os import path, getcwd
from glob import glob
import csv



def cube():
    """
    Fonction qui output les rotations d'objet nécessaires pour obtenir les points de vue 
    de notre objet 3D depuis les faces d'un cube
    @return viewpoints : list [trimesh.transformations.rotation_matrix] : liste des matrices
    de rotation depuis l'axe x, position (0,0,0).
    """
    viewpoints = [
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[0, 1, 0]),
        trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[1, 0, 0]),
        trimesh.transformations.rotation_matrix(angle=-np.pi/2, direction=[1, 0, 0]),
    ]
    return viewpoints

def icosahedron():
    """
    Fonction qui output les rotations d'objet nécessaires pour obtenir les points de vue 
    de notre objet 3D depuis les sommets d'un isocahèdre (la forme d'un dé 20, avec 12 sommets), 
    ceci nous permet de ne plus avoir de vue de dessous ou de dessus comme c'était proposé avec
    la solution du cube, puisque ce sont les vue déja générées par le modèle Multi Vu, ainsi
    on évalue réellement la capacité du modèle 3D à reconstituer de nouvelles vues
    sans stable diffusion...
    @return viewpoints : list [trimesh.transformations.rotation_matrix] : liste des matrices
    de rotation depuis l'axe x, position (0,0,0).
    """
    #définition des sommets de notre icosahedre
    phi = (1+np.sqrt(5))/2
    vertices = [[phi, 1, 0], [phi, -1, 0], [-phi, -1, 0], [-phi, 1, 0],
                [1, 0, phi], [-1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
                [0, phi, 1], [0, phi, -1], [0, -phi, -1], [0, -phi, 1]]
    vertices = [np.array(e) for e in vertices]
    #vertices provided in the forum https://math.stackexchange.com/questions/2174594/co-ordinates-of-the-vertices-an-icosahedron-relative-to-its-centroid
    X = np.array([1,0,0]) #point de vue initial
    O = np.array([0,0,0]) #centre de la mesh
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


# Fonctions pour le nettoyage de texte et la similarité
def get_cleaned_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower().strip()
    formatted_text = ' '.join(cleaned_text.split())
    return formatted_text

def dot_product(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

def magnitude(vector):
    return math.sqrt(sum(x**2 for x in vector))

def cosine_similarity(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    mag1 = magnitude(vector1)
    mag2 = magnitude(vector2)
    return dot_prod / (mag1 * mag2)

vectorizer = TfidfVectorizer()
def get_tfidf_matrix(documents):
    return vectorizer.fit_transform(documents)

def compare_two_prompts(prompt, prompts):
    cleaned_texts = [get_cleaned_text(prompt)] + [get_cleaned_text(p) for p in prompts]
    tfidf_matrix = get_tfidf_matrix(cleaned_texts)
    prompt_vector = tfidf_matrix.getrow(0).toarray()[0]
    scores = []
    for i in range(1, len(prompts) + 1):
        vector_i = tfidf_matrix.getrow(i).toarray()[0]
        score = cosine_similarity(prompt_vector, vector_i)
        scores.append(score)
    mean_score = sum(scores) / len(scores)
    return mean_score, scores

def screenshot_the_mesh(mesh, out_folder='./imagesout_01/', method=cube, save_views=False):
    resolution = (512, 384)
    filename_format = "{:03d}.png"
    scene = mesh.scene()
    viewpoints = method()
    screenshots = []
    for i, viewpoint in enumerate(viewpoints):
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        png = scene.save_image(resolution=resolution, visible=True)
        img_arr = cv2.imdecode(np.fromstring(png, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        screenshots.append(rgb_img)
        if save_views:
            with open(path.join(out_folder, filename_format.format(i)), "wb") as f:
                f.write(png)
        scene.graph[scene.camera.name] = camera_old
    return screenshots

def prompts_the_views(processor, model, screenshots):
    prompts = []
    for raw_image in screenshots:
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        prompts.append(processor.decode(out[0], skip_special_tokens=True))
    return prompts
def main(method, folder_mother, name, save_views):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    cwd = getcwd()
    folder_mother = path.relpath(folder_mother, cwd)
    with open(path.join(folder_mother, 'prompts.txt'), 'r') as f:
        prompts = [line.strip()[:] for line in f.readlines()]

    scores = []
    for n_prompt, folder in enumerate(tqdm(sorted(glob(path.join(folder_mother, '[0-9]*/'))))):
        folder_name = path.relpath(folder, cwd)
        asset_path = path.join(folder_name, name + '.obj') if name else glob(path.join(folder, "*.obj"))[0]
        mesh = trimesh.load(asset_path)
        screenshots = screenshot_the_mesh(mesh=mesh, out_folder=folder, method=method, save_views=save_views)
        prompts_created = prompts_the_views(processor, model, screenshots)
        prompt_main = prompts[n_prompt]
        mean_score, individual_scores = compare_two_prompts(prompt_main, prompts_created)
        scores.append(mean_score)
        print(f'Prompt: {prompt_main} - Average Score: {mean_score:.3f}')
        for i, score in enumerate(individual_scores):
            print(f'\tPrompt {i+1}: {prompts_created[i]} - Score: {score:.3f}')

    print(f"The overall average score is {np.mean(scores):.3f}")
def main(method, folder_mother, name, save_views):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model_comparation = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    cwd = getcwd()
    folder_mother = path.relpath(folder_mother, cwd)
    with open(path.join(folder_mother, 'prompts.txt'), 'r') as f:
        prompts = [line.strip()[:] for line in f.readlines()]
    #loop on each asset
    scores = []
    for n_prompt,folder in enumerate(tqdm(sorted(glob(path.join(folder_mother, '[0-9]*/'))))):
        

        folder_name = path.relpath(folder, cwd)

        #manipulationen fonction du nom de la mesh
        if name:
            asset_path = path.join(folder_name, name +'.obj')
            if not path.isfile(asset_path):
                raise FileNotFoundError(f"Le dossier '{folder_name}' est incomplet, il devrait contenir : '{name}.obj'")
            name_csv = '/' + name + '_scores.csv'
            out_message = f', pour le model {name}'
        else:
            asset_path = glob(path.join(folder, "*.obj"))
            if len(asset_path!=1):
                raise FileNotFoundError(f"Le dossier '{folder_name}' contient plus d'un fichier .obj, il ne doit en contenir qu'un seul exactement")
            name_csv = '/scores.csv'
            out_message = ''
        mesh = trimesh.load(asset_path)
        screenshots = screenshot_the_mesh(mesh = mesh, out_folder = folder, method = method, save_views = save_views)
        prompts_created = prompts_the_views(processor = processor, model = model, screenshots = screenshots)
        prompt_main = prompts[n_prompt]
        score = compare_two_prompts(prompt_main,prompts_created, model_comparation)
        scores.append(score)
        with open(folder_mother + name_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([n_prompt, score, prompts[n_prompt] ])
    out_message = f"le score moyen que nous venons de calculer est de {np.mean(scores)}" + out_message
    print(out_message)

list_to_delete = ["with a white background", "on a white background"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the performance of my 3D assets')
    parser.add_argument("--method", help="The method to use for views (cube or icosahedron)", default=icosahedron)
    parser.add_argument("--folder_mother", help="Directory containing the 3D mesh files", default='./', type=str)
    parser.add_argument("--name", help="Name of the .obj files", default=None, type=str)
    parser.add_argument("--save_views", help="Whether to save views of the model", default=False, type=bool)
    args = parser.parse_args()
    main(**vars(args))
