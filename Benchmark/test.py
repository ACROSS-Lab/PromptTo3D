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
import glob
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('wordnet')
nltk.download('stopwords')
import os



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
def preprocess_text(text):
    # Normalize text: convert to lowercase and remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text.lower()).strip()
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    
    return ' '.join(lemmatized_tokens)


# Configure the vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english')


def get_tfidf_cosine_scores(reference_prompt, generated_prompts):
    vectorizer = TfidfVectorizer()
    documents = [reference_prompt] + generated_prompts
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_scores = []
    ref_vector = tfidf_matrix[0].toarray()
    for i in range(1, len(documents)):
        gen_vector = tfidf_matrix[i].toarray()
        # Compute cosine similarity and extract the scalar value immediately
        score = cosine_similarity(ref_vector, gen_vector)[0][0]
        cosine_scores.append(score)
    return cosine_scores

def screenshot_the_mesh(mesh, out_folder='./imagesout_01/', method=None, save_views=False):
    resolution = (512, 384)
    filename_format = "{:03d}.png"
    scene = mesh.scene()
    viewpoints = method() if method else []
    screenshots = []

    for i, viewpoint in enumerate(viewpoints):
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        png = scene.save_image(resolution=resolution, visible=True)
        img_arr = np.frombuffer(png, dtype=np.uint8)  
        rgb_img = cv2.cvtColor(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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
    cwd = os.getcwd()
    folder_mother = os.path.relpath(folder_mother, cwd)
    prompts_path = os.path.join(folder_mother, 'prompts.txt')
    
    if not os.path.exists(prompts_path):
        print("Error: No reference prompts loaded. Check the 'prompts.txt' file.")
        return
    
    with open(prompts_path, 'r') as f:
        reference_prompts = [line.strip() for line in f.readlines()]

    directory_paths = sorted(glob.glob(os.path.join(folder_mother, '[0-9]*/')))
    if len(reference_prompts) != len(directory_paths):
        print(f"Warning: The number of reference prompts ({len(reference_prompts)}) does not match the number of directories ({len(directory_paths)}).")
    
    scores = []

    for n_prompt, folder in enumerate(tqdm(directory_paths)):
        if n_prompt >= len(reference_prompts) or not reference_prompts[n_prompt]:
            print(f"Skipping empty or nonexistent reference prompt at index {n_prompt}.")
            continue
        
        folder_name = os.path.relpath(folder, cwd)
        asset_path = os.path.join(folder_name, name + '.obj') if name else glob.glob(os.path.join(folder, "*.obj"))[0]
        mesh = trimesh.load(asset_path)
        screenshots = screenshot_the_mesh(mesh=mesh, out_folder=folder, method=method, save_views=save_views)
        generated_prompts = prompts_the_views(processor, model, screenshots)
        individual_scores = get_tfidf_cosine_scores(reference_prompts[n_prompt], generated_prompts)
        
        # Debug print to check if individual_scores is not empty
        print(f"Debug - Individual Scores: {individual_scores}")

        scores.extend(individual_scores) 
        # Debug print to see the scores list after extension
        print(f"Debug - Scores List: {scores}")

        
        for generated_prompt, score in zip(generated_prompts, individual_scores):
            # Debugging the type and structure of score
            print(f"Debug - Type of score: {type(score)}, Score: {score}")
            if isinstance(score, np.ndarray):
                score = score.item()  # Convert ndarray to scalar if necessary
            print(f'Reference Prompt: "{reference_prompts[n_prompt]}"\nGenerated Prompt: "{generated_prompt}"\nScore: {score:.3f}')




    if scores:
        average_score = np.mean(scores)
        print(f"The overall average score is {average_score:.3f}")
    else:
        print("No scores were calculated.")
    average_percentage = average_score * 100  # Convert to percentage
    print(f"The overall average score in percentis {average_percentage:.2f}%")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the performance of 3D assets using different viewing methods.')
    parser.add_argument("--method", type=str, choices=['cube', 'icosahedron'], default='icosahedron',
                        help="Method to use for generating views of the model. Options: 'cube' or 'icosahedron'.")
    parser.add_argument("--folder_mother", type=str, default='./', help="Directory containing the 3D mesh files.")
    parser.add_argument("--name", type=str, help="Optional specific name of the .obj files to use.")
    parser.add_argument("--save_views", type=bool, default=False, help="Whether to save generated views of the model.")
    args = parser.parse_args()

    # Mapping string method names to function calls dynamically
    method_function = globals().get(args.method)
    if not method_function:
        print(f"Error: Method {args.method} not found.")
    else:
        main(method_function, args.folder_mother, args.name, args.save_views)

