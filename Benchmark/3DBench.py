import trimesh
import numpy as np
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from glob import glob
from os import path, getcwd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse
import cv2
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

def screenshot_the_mesh(mesh, out_folder ='./imagesout_01/', method = icosahedron, save_views = False):
    ##### Tout d'abord la partie Screenshot  ########
    resolution=(512, 384)
    filename_format="{:03d}.png"
    scene = mesh.scene()
    #On considère l'icosahedre
    viewpoints = method()
    screenshots = [] #generator ????
    for i, viewpoint in enumerate(viewpoints):
        # bouger la camera
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        # sauvegarder les images, nécessaire ????
        filename = filename_format.format(i)
        try :
            png = scene.save_image(resolution=resolution, visible=True)
            img_arr = cv2.imdecode(np.fromstring(png, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) 
            screenshots.append(rgb_img)
            if save_views :
                with open(path.join(out_folder,filename), "wb") as f:
                    f.write(png)
            # repartir de la vue initiale..
            scene.graph[scene.camera.name] = camera_old
        except ZeroDivisionError:
            print("Error: Window resizing caused division by zero. Try setting minimum window size or handling resizing events.")
    return screenshots

def prompts_the_views(processor, model, screenshots):
    Prompts = []
    #for img in glob(path.relpath(out_dir+'/*.png')):
        #raw_image = Image.open(img).convert('RGB')
    for raw_image in screenshots:
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        Prompts.append(processor.decode(out[0], skip_special_tokens=True))
    return Prompts


query = "a dog eating a slice of watermelon"
docs = ["dog eats watermelon", "a dog sleeping", "a couch whith a watermelon on it", "a motorcycle", "a corgi devouring a piece of melon", "a pitt-bull eating a fruit"]

def compare_two_prompts(prompt, prompts, model_comparation, show_scores = False):
    prompt_embedded = model_comparation.encode(prompt)
    prompts_embedded = model_comparation.encode(prompts)
    scores = util.dot_score(prompt_embedded, prompts_embedded)[0].cpu().tolist()
    if show_scores :
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for doc, score in doc_score_pairs:
            print(score, doc)
    mean_score = sum(scores)/len(scores)
    return mean_score

def main(method, folder_mother, name, save_views):
    ##Load the models that will be used
    #First the image to prompt model 
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    #load the model that will compare the prompts
    model_comparation = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    cwd = getcwd()
    with open(path.join(folder_mother, '*.txt'), 'r') as f:
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
            out_message = f',pour le model {name}'
        else:
            asset_path = glob(path.join(folder, "*.obj"))
            if len(asset_path!=1):
                raise FileNotFoundError(f"Le dossier '{folder_name}' contient plus d'un fichier .obj, il ne doit en contenir qu'un seul exactement")
            name_csv = '/scores.csv'
            out_message = ''
        mesh = trimesh.load(asset_path)
        screenshots = screenshot_the_mesh(mesh = mesh, method = method, save_views = save_views)
        prompts_created = prompts_the_views(processor = processor, model = model, screenshots = screenshots)
        prompt_main = prompts[n_prompt]
        score = compare_two_prompts(prompt_main,prompts_created, model_comparation)
        scores.append(score)
        with open(folder_mother + name_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([n_prompt, score, prompts[n_prompt] ])
    out_message = f"le score moyen que nous venons de calculer est de {np.mean(scores)}" + out_message
    print(out_message)




if __name__== "__main__" :
    parser = argparse.ArgumentParser(description = 'Evaluer les performances de mes assets 3D')
    parser.add_argument(
        "--method",
        help = "vues prises, que ce soit celles d'un icosahedron (12 sommets) ou d'un cube (6 faces)...",
        default = icosahedron
    )
    parser.add_argument(
        "--folder_mother",
        help = "fichier dans lequel sont situees les mesh de nos objets 3D",
        default = '.',
        type = str
    )
    parser.add_argument(
        "--name",
        help = "nom des fichiers .obj",
        default = None,
        type = str
    )
    parser.add_argument(
        "--save_views",
        default = False,
        type = bool,
        help = "booléen qui décide si oui ou non on sauvegarde les vues de notre modèle"


    )
    args = parser.parse_args()
    main(*vars(args))