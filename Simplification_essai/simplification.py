import argparse
import os

from Mesh import Mesh  # Assurez-vous que le module Mesh est bien disponible et correctement défini.

def get_parser():
    # Création et configuration du parseur d'arguments pour le script.
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file name")  # Argument pour le fichier d'entrée.
    parser.add_argument("-v", type=int, help="Target vertex number")  # Nombre cible de sommets.
    parser.add_argument("-p", type=float, default=0.5, help="Rate of simplification (Ignored by -v)")  # Taux de simplification.
    parser.add_argument("-optim", action="store_true", help="Specify for valence aware simplification")  # Simplification sensible à la valence.
    parser.add_argument("-isotropic", action="store_true", help="Specify for Isotropic simplification")  # Simplification isotrope.
    return parser.parse_args()

def display_mesh_info(mesh, prefix=""):
    # Affiche et renvoie des informations sur le maillage : nombre de sommets et de polygones.
    num_vertices = len(mesh.vs)
    num_faces = len(mesh.fa)
    print(f"{prefix} Number of vertices: {num_vertices}")
    print(f"{prefix} Number of polygons: {num_faces}")
    return num_vertices, num_faces

def main():
    args = get_parser()  # Récupération des arguments de la ligne de commande.
    mesh = Mesh(args.input)  # Chargement du maillage à partir du fichier spécifié.
    input_vertices, input_faces = display_mesh_info(mesh, prefix="Input Mesh:")  # Affichage des infos du maillage d'entrée.

    mesh_name = os.path.basename(args.input).split(".")[-2]  # Extraction du nom de base du fichier pour le sauvegarder plus tard.
    if args.v:
        target_v = args.v  # Utilisation du nombre de sommets cible spécifié.
    else:
        target_v = int(len(mesh.vs) * args.p)  # Calcul du nombre de sommets cible basé sur le pourcentage.
    if target_v >= mesh.vs.shape[0]:
        print("[ERROR]: Target vertex number should be smaller than {}!".format(mesh.vs.shape[0]))  # Vérification que le nombre cible est valide.
        exit()
    if args.isotropic:
        simp_mesh = mesh.edge_based_simplification(target_v=target_v, valence_aware=args.optim)  # Simplification isotrope si spécifiée.
    else:
        simp_mesh = mesh.simplification(target_v=target_v, valence_aware=args.optim)  # Simplification standard.

    output_vertices, output_faces = display_mesh_info(simp_mesh, prefix="Simplified Mesh:")  # Affichage des infos du maillage simplifié.

    os.makedirs("data/output/", exist_ok=True)  # Création du dossier de sortie si nécessaire.
    simp_mesh.save(f"data/output/{mesh_name}_{simp_mesh.vs.shape[0]}.obj")  # Sauvegarde du maillage simplifié.
    print(f"[FIN] Simplification Completed! Reduction in vertices: {input_vertices - output_vertices}, Reduction in polygons: {input_faces - output_faces}")  # Rapport final.

if __name__ == "__main__":
    main()  
