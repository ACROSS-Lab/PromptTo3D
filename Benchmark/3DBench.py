import trimesh
import numpy as np


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

mesh = trimesh.load("model1.obj")
prompt = ""

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

def screenshot_the_mesh(mesh, prompt, method = icosahedron):
    ##### Tout d'abord la partie Screenshot  ########
    resolution=(512, 384)
    filename_format="view_{:03d}.png"
    scene = mesh.scene()
    #On considère l'icosahedre
    viewpoints = method()
    for i, viewpoint in enumerate(viewpoints):
        # bouger la camera
        camera_old, _ = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        # sauvegarder les images, nécessaire ????
        filename = filename_format.format(i)
        try :
            png = scene.save_image(resolution=resolution, visible=True)
            with open(filename, "wb") as f:
                f.write(png)
            # repartir de la vue initiale..
            scene.graph[scene.camera.name] = camera_old
            print(f"vue n {i} enregistrée")
        except ZeroDivisionError:
            print("Error: Window resizing caused division by zero. Try setting minimum window size or handling resizing events.")



screenshot_the_mesh(mesh, prompt)
