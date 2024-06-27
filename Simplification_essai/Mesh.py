import numpy as np
import scipy as sp
import heapq
import copy
from tqdm import tqdm
from sklearn.preprocessing import normalize

OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path, build_code=False, build_mat=False, manifold=True):
        self.path = path
        self.vs, self.faces = self.fill_from_file(path)
        self.compute_face_normals()
        self.compute_face_center()
        self.device = 'cpu'
        self.simp = False
        
        if manifold:
            self.build_gemm() #self.edges, self.ve
            self.compute_vert_normals()
            self.build_v2v()
            self.build_vf()
            self.build_uni_lap()
    
    def fill_from_file(self, path):

        '''
        Lit un fichier pour extraire des informations de géométrie, typiquement utilisée 
        pour les fichiers de modèle 3D (comme ceux au format OBJ). Elle gère deux types de données : les sommets (v) et les faces (f).
        '''
        vs, faces = [], []  # Initialisation de deux listes pour stocker les sommets et les indices des faces.
        f = open(path)  
        for line in f:  # Boucle sur chaque ligne du fichier.
            line = line.strip()  # Supprime les espaces blancs 
            splitted_line = line.split()  
            if not splitted_line:  # Continue à la prochaine itération si la ligne est vide.
                continue
            elif splitted_line[0] == 'v':  # Si la ligne débute par 'v', elle décrit un sommet.
                vs.append([float(v) for v in splitted_line[1:4]])  # Convertit les trois prochains éléments en flottants et les ajoute à la liste des sommets.
            elif splitted_line[0] == 'f':  # Si la ligne débute par 'f', elle décrit une face.
                # Extrait les indices des sommets de la face, en gérant les éventuelles références aux coordonnées de texture ou aux normales (séparées par '/').
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                # Assure que chaque face est un triangle.
                assert len(face_vertex_ids) == 3
                # Ajuste les indices basés sur le fait que certains formats peuvent utiliser des indices négatifs ou commencer à compter à partir de 1.
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)  # Ajoute les indices ajustés à la liste des faces.
        f.close()  # Ferme le fichier.
        vs = np.asarray(vs)  # Convertit la liste des sommets en un tableau numpy.
        faces = np.asarray(faces, dtype=int)  # Convertit la liste des indices des faces en un tableau numpy de type entier.

        # Vérifie que tous les indices de faces sont valides.
        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces  # Retourne les tableaux des sommets et des faces.
    def build_gemm(self):
        # Initialisation des listes pour stocker les informations relatives aux sommets et arêtes.
        self.ve = [[] for _ in self.vs]  # Liste de listes pour les arêtes connectées à chaque sommet.
        self.vei = [[] for _ in self.vs]  # Liste de listes pour les indices des sommets dans les arêtes.
        edge_nb = []  # Liste pour stocker les arêtes voisines de chaque arête.
        sides = []  # Liste pour stocker les côtés correspondants des arêtes voisines.
        edge2key = dict()  # Dictionnaire pour mapper les arêtes à une clé unique.
        edges = []  # Liste pour stocker les arêtes.
        edges_count = 0  # Compteur pour attribuer un indice unique à chaque arête.
        nb_count = []  # Liste pour compter le nombre de voisins de chaque arête.

        # Boucle pour traiter chaque face du maillage.
        for face_id, face in enumerate(self.faces):
            faces_edges = []  # Liste pour stocker temporairement les arêtes de la face actuelle.
            for i in range(3):  # Itération sur les trois sommets de chaque face (triangle).
                cur_edge = (face[i], face[(i + 1) % 3])  # Formation d'une arête entre sommets consécutifs.
                faces_edges.append(cur_edge)  # Ajout de l'arête à la liste temporaire.

            # Traitement des arêtes pour les normaliser et les enregistrer.
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(edge))  # Normalisation de l'arête (ordre croissant des sommets).
                faces_edges[idx] = edge  # Mise à jour de l'arête dans la liste.
                if edge not in edge2key:  # Vérification de l'existence préalable de l'arête.
                    edge2key[edge] = edges_count  # Attribution d'un indice unique à l'arête.
                    edges.append(list(edge))  # Ajout de l'arête à la liste des arêtes.
                    edge_nb.append([-1, -1, -1, -1])  # Initialisation des voisins de l'arête.
                    sides.append([-1, -1, -1, -1])  # Initialisation des côtés des voisins.
                    self.ve[edge[0]].append(edges_count)  # Enregistrement de l'arête pour le premier sommet.
                    self.ve[edge[1]].append(edges_count)  # Enregistrement de l'arête pour le second sommet.
                    self.vei[edge[0]].append(0)  # Enregistrement de l'indice du premier sommet dans l'arête.
                    self.vei[edge[1]].append(1)  # Enregistrement de l'indice du second sommet dans l'arête.
                    nb_count.append(0)  # Initialisation du compteur de voisins pour cette arête.
                    edges_count += 1  # Incrémentation du compteur d'arêtes.

            # Enregistrement des arêtes voisines et mise à jour des comptes et côtés correspondants.
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]  # Récupération de l'indice unique de l'arête.
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]  # Enregistrement du premier voisin.
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]  # Enregistrement du second voisin.
                nb_count[edge_key] += 2  # Mise à jour du compteur de voisins.
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]  # Récupération de l'indice unique de l'arête.
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1 # Mise à jour du voisin du premier sommet.
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2 - 2  # Mise à jour du côté du second voisin.

        # Conversion des listes en tableaux numpy pour une utilisation plus efficace.
        self.edges = np.array(edges, dtype=np.int32)  # Tableau des arêtes.
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)  # Tableau des arêtes voisines.
        self.sides = np.array(sides, dtype=np.int64)  # Tableau des côtés des arêtes voisines.
        self.edges_count = edges_count  # Stockage du nombre total d'arêtes.
    def compute_face_normals(self):
    # Calcul des vecteurs normaux de chaque face via produit vectoriel.
        face_normals = np.cross(
            self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], 
            self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]]
        )

        # Normalisation des normales de face pour éviter les divisions par zéro.
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24

        # Calcul de l'aire des faces en utilisant la longueur des normales.
        face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))

        # Division des normales par leur norme pour obtenir des vecteurs unitaires.
        face_normals /= norm

        # Assignation des normales et aires calculées aux attributs de l'instance.
        self.fn, self.fa = face_normals, face_areas
    def compute_vert_normals(self):
        # Initialisation du tableau de normales des sommets avec des zéros.
        vert_normals = np.zeros((3, len(self.vs)))

        # Récupération des normales de face précalculées.
        face_normals = self.fn
        # Récupération des indices des faces.
        faces = self.faces

        # Nombre de sommets et de faces.
        nv = len(self.vs)
        nf = len(faces)

        # Création d'une matrice d'incidence face-à-sommet.
        # Chaque ligne correspond à un sommet et chaque colonne à une face.
        # 'mat_rows' contient les indices des sommets répétés pour chaque face.
        mat_rows = faces.reshape(-1)
        # 'mat_cols' contient les indices des faces répétés pour chaque sommet de la face.
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        # 'mat_vals' est un tableau de 1, indiquant la présence d'un sommet dans une face.
        mat_vals = np.ones(len(mat_rows))

        # Construction de la matrice creuse à partir des triplets (valeurs, indices des lignes, indices des colonnes).
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))

        # Multiplication de la matrice d'incidence par les normales de face pour accumuler les normales de face sur les sommets correspondants.
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)

        # Normalisation des normales de sommet pour en faire des vecteurs unitaires.
        vert_normals = normalize(vert_normals, norm='l2', axis=1)

        # Stockage des normales de sommets normalisées dans l'attribut de l'instance.
        self.vn = vert_normals

    def compute_face_center(self):
    # Récupération des indices des faces et des coordonnées des sommets.
        faces = self.faces
        vs = self.vs
        # Calcul du centre de chaque face.
        # On accède aux coordonnées des sommets de chaque face, on les somme et on divise par 3 (pour les triangles).
        self.fc = np.sum(vs[faces], axis=1) / 3.0
    def build_uni_lap(self):
        """compute uniform laplacian matrix"""
        # Récupérer les arêtes et les listes de sommets connectés pour chaque sommet.
        edges = self.edges
        ve = self.ve

        # Calculer les voisins de chaque sommet en excluant le sommet lui-même.
        # 'sub_mesh_vv' contient les indices des sommets voisins pour chaque sommet.
        sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
        sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

        # Nombre de sommets dans le maillage.
        num_verts = self.vs.shape[0]

        # Construction des indices de lignes pour la matrice creuse.
        # Chaque sommet i apparaît len(vv) fois, où vv est l'ensemble de ses voisins.
        mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]
        mat_rows = np.concatenate(mat_rows)

        # Construction des indices de colonnes pour la matrice creuse.
        # Chaque entrée correspond à un voisin du sommet i.
        mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
        mat_cols = np.concatenate(mat_cols)

        # Valeurs pour les entrées non-diagonales de la matrice, marquant les connexions entre voisins.
        mat_vals = np.ones_like(mat_rows, dtype=np.float32) * -1.0

        # Construction d'une matrice creuse pour les connexions entre les sommets et leurs voisins.
        neig_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))

        # Calcul de la somme des connexions pour chaque sommet, utilisé pour la diagonale.
        sum_count = sp.sparse.csr_matrix.dot(neig_mat, np.ones((num_verts, 1), dtype=np.float32))

        # Indices de ligne et de colonne pour les entrées diagonales.
        mat_rows_ident = np.array([i for i in range(num_verts)])
        mat_cols_ident = np.array([i for i in range(num_verts)])
        # Les entrées diagonales sont les négatifs de la somme des connexions pour chaque sommet.
        mat_ident = np.array([-s for s in sum_count[:, 0]])

        # Combinaison des indices et valeurs pour les parties diagonales et non diagonales.
        mat_rows = np.concatenate([mat_rows, mat_rows_ident], axis=0)
        mat_cols = np.concatenate([mat_cols, mat_cols_ident], axis=0)
        mat_vals = np.concatenate([mat_vals, mat_ident], axis=0)

        # Construction finale de la matrice de Laplacien.
        self.lapmat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))
    def build_vf(self):
    # Créer une liste d'ensembles pour chaque sommet pour stocker les indices des faces connectées.
        vf = [set() for _ in range(len(self.vs))]

        # Remplir les ensembles avec les indices des faces pour chaque sommet.
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)

        # Attribuer la liste construite à l'attribut d'instance 'vf'.
        self.vf = vf
    def build_v2v(self):
    # Initialiser une liste de voisins pour chaque sommet.
        v2v = [[] for _ in range(len(self.vs))]

        # Peupler la liste de voisins pour chaque sommet en utilisant les arêtes.
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])

        # Sauvegarder la liste de voisins dans l'attribut de classe.
        self.v2v = v2v

        # Créer une matrice d'adjacence pour les sommets.
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1).astype(np.int64)
        v2v_vals = np.ones(v2v_inds.shape[1], dtype=np.float32)
        self.v2v_mat = sp.sparse.csr_matrix((v2v_vals, v2v_inds), shape=(len(self.vs), len(self.vs)))

        # Calculer le degré de chaque sommet.
        self.v_dims = np.sum(self.v2v_mat.toarray(), axis=1)
    def simplification(self, target_v, valence_aware=True, midpoint=False):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        """ 1. compute Q for each vertex """
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = np.array(list(vf[i]))
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1

            if midpoint:
                v_new = 0.5 * (v_0 + v_1)
                v4_new = np.concatenate([v_new, np.array([1])])
            else:
                Q_lp = np.eye(4)
                Q_lp[:3] = Q_new[:3]
                try:
                    Q_lp_inv = np.linalg.inv(Q_lp)
                    v4_new = np.matmul(Q_lp_inv, np.array([[0,0,0,1]]).reshape(-1,1)).reshape(-1)
                except:
                    v_new = 0.5 * (v_0 + v_1)
                    v4_new = np.concatenate([v_new, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = vf[e[0]].intersection(vf[e[1]])
                valence_new = len(vf[e[0]].union(vf[e[1]]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)
            
            
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (e[0], e[1])))
        
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]
        pbar = tqdm(total=np.sum(vi_mask)-target_v, desc="Processing")
        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                self.remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap)
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware=valence_aware)
                pbar.update(1)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh
    def edge_based_simplification(self, target_v, valence_aware=True):
    # Extraction des données de maillage nécessaires.
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        # Calcul de la longueur de chaque arête.
        edge_len = vs[edges][:,0,:] - vs[edges][:,1,:]
        edge_len = np.linalg.norm(edge_len, axis=1)
        edge_len_heap = np.stack([edge_len, np.arange(len(edge_len))], axis=1).tolist()
        heapq.heapify(edge_len_heap)

        # Préparation d'un tas pour les longueurs d'arête, pour simplifier le maillage en commençant par les plus courtes.
        E_heap = []
        for i, e in enumerate(edges):
            heapq.heappush(E_heap, (edge_len[i], (e[0], e[1])))

        # Création d'une copie du maillage pour effectuer les modifications.
        simp_mesh = copy.deepcopy(self)
        vi_mask = np.ones([len(simp_mesh.vs)], dtype=bool)
        fi_mask = np.ones([len(simp_mesh.faces)], dtype=bool)
        vert_map = [{i} for i in range(len(simp_mesh.vs))]
        pbar = tqdm(total=np.sum(vi_mask) - target_v, desc="Processing")

        # Boucle principale de simplification, effondrement des arêtes jusqu'à atteindre le nombre de sommets cible.
        while np.sum(vi_mask) > target_v:
            if not E_heap:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            # Extraction de l'arête avec la longueur minimale.
            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            # Vérification si les sommets sont toujours actifs avant de procéder.
            if not (vi_mask[vi_0] and vi_mask[vi_1]):
                continue

            # Déterminer les voisins partagés et les faces fusionnées pour évaluer si l'effondrement est possible.
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            # Conditions pour s'assurer que l'effondrement ne crée pas de géométrie non-manifold ou ne supprime pas de bordures.
            if len(shared_vv) != 2 or len(merged_faces) != 2:
                continue

            # Effondrement de l'arête, mise à jour des structures de données et recomptage.
            self.edge_based_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware)
            pbar.update(1)

        # Reconstruction du maillage simplifié et finalisation.
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)

        return simp_mesh
    @staticmethod
    def remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap):
    # Placeholder pour le traitement spécifique des situations non-manifold lors de l'effondrement des arêtes.
        pass  

    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        """ recompute E """
        Q_0 = Q_s[vi_0]
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            Q_1 = Q_s[vv_i]
            Q_new = Q_0 + Q_1
            v4_mid = np.concatenate([v_mid, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)

            E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))
    

    def edge_based_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware):
        # Identifie les sommets partagés et définit les nouveaux voisins après l'effondrement.
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})

        # Fusionne les listes de faces des deux sommets et supprime les faces qui ne sont plus valides.
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        # Met à jour les connexions des sommets voisins pour refléter le nouveau sommet.
        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        # Met à jour le mappage des sommets pour enregistrer les modifications.
        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()

        # Marque les faces impliquées dans l'effondrement comme non valides.
        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        # Ajuste la position du sommet résultant à la moyenne des deux sommets effondrés.
        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        # Recalcule les métriques pour l'heap, ajustant pour les changements dus à l'effondrement.
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            edge_len = np.linalg.norm(simp_mesh.vs[vi_0] - simp_mesh.vs[vv_i])
            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)
                edge_len *= valence_penalty

            heapq.heappush(E_heap, (edge_len, (vi_0, vv_i)))
    @staticmethod

    def valence_weight(valence_new):
    # Calculer la pénalité basée sur la différence entre la valence actuelle et une valence optimale.
        valence_penalty = abs(valence_new - OPTIM_VALENCE) * VALENCE_WEIGHT + 1

        # Si la valence est exactement 3, appliquer une pénalité très élevée pour décourager cette configuration.
        if valence_new == 3:
            valence_penalty *= 100000

        return valence_penalty
    @staticmethod

    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
    # Créer un mappage de l'ancien indice de sommet vers le nouvel indice basé sur les sommets actifs.
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        
        # Filtrer les sommets du maillage en utilisant vi_mask pour garder seulement les sommets actifs.
        simp_mesh.vs = simp_mesh.vs[vi_mask]
        
        # Créer un dictionnaire pour mappage des sommets selon vert_map, qui contient des groupes de sommets fusionnés.
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i

        # Remapper les indices des sommets dans les faces pour refléter les fusions de sommets.
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        # Filtrer les faces en utilisant fi_mask pour éliminer les faces supprimées.
        simp_mesh.faces = simp_mesh.faces[fi_mask]

        # Remapper les indices de sommets dans les faces pour utiliser les indices des sommets actifs.
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        # Recalculer les normales des faces, les centres des faces, et reconstruire les autres structures nécessaires.
        simp_mesh.compute_face_normals()
        simp_mesh.compute_face_center()
        simp_mesh.build_gemm()
        simp_mesh.compute_vert_normals()
        simp_mesh.build_v2v()
        simp_mesh.build_vf()
    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
    # Initialiser les dictionnaires pour le mappage de hachage.
        pool_hash = {}
        unpool_hash = {}

        # Construire le hachage en parcourant les indices des sommets actifs.
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            # Vérifier que chaque sommet actif a un mappage valide dans vert_map.
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            
            # Pour chaque sommet original dans vert_map, associer le nouvel indice de sommet simplifié.
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        # Vérifier l'intégrité des mappages pour s'assurer que tous les sommets originaux sont couverts.
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)
        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vetices cannot be covered!")
            return
        
        # Trier le hachage pour l'ordre constant des accès et sauvegarder dans les attributs du maillage.
        pool_hash = sorted(pool_hash.items(), key=lambda x: x[0])
        simp_mesh.pool_hash = pool_hash
        simp_mesh.unpool_hash = unpool_hash
    def save(self, filename):
    # Assurer que le maillage contient des sommets avant de sauvegarder.
        assert len(self.vs) > 0

        # Préparer les données des sommets et des indices des faces pour l'écriture.
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        # Ouvrir le fichier spécifié pour l'écriture des données du maillage.
        with open(filename, 'w') as fp:
            # Écrire les coordonnées de chaque sommet en utilisant le format OBJ.
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Écrire les indices des faces, en notant que le format OBJ commence les indices à 1.
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
    def save_as_ply(self, filename, fn):
    # Vérifier que le maillage contient des sommets.
        assert len(self.vs) > 0

        # Préparer les données des sommets, des indices de faces et des normales de faces.
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()
        fnormals = np.array(fn, dtype=np.float32).flatten()

        # Ouvrir le fichier en mode écriture.
        with open(filename, 'w') as fp:
            # Écrire l'en-tête PLY spécifiant le format et les propriétés des éléments.
            fp.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(self.vs)))
            fp.write("property float x\nproperty float y\nproperty float z\n")
            fp.write("element face {}\n".format(len(self.faces)))
            fp.write("property list uchar int vertex_indices\n")
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
            fp.write("end_header\n")
            
            # Écrire les coordonnées de chaque sommet.
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write("{0:.6f} {1:.6f} {2:.6f}\n".format(x, y, z))
            
            # Écrire les indices des faces avec des valeurs de couleur codées basées sur les normales.
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0]
                i1 = indices[i + 1]
                i2 = indices[i + 2]
                c0 = fnormals[i + 0]
                c1 = fnormals[i + 1]
                c2 = fnormals[i + 2]
                c0 = np.clip(int(255 * c0), 0, 255)
                c1 = np.clip(int(255 * c1), 0, 255)
                c2 = np.clip(int(255 * c2), 0, 255)
                c3 = 255  # Alpha value
                fp.write("3 {0} {1} {2} {3} {4} {5} {6}\n".format(i0, i1, i2, c0, c1, c2, c3))
    
    def pool_main(self, target_v):
    # Load all necessary mesh data structures for operation.
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        # Initialize matrices for storing quadric errors and error scalars for each vertex.
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            # Fetch indices of faces associated with current vertex.
            f_s = np.array(list(vf[i]))
            # Extract face centers and normals corresponding to these faces.
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            # Compute the distance (d) term used in the quadric error matrices.
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            # Combine face normals and distance terms to form complete quadric data.
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            # Calculate the quadric error matrix for the current vertex.
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            # Prepare a homogeneous coordinate for the vertex.
            v4 = np.concatenate([v, np.array([1])])
            # Compute the initial error scalar for the current vertex.
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        # Initialize a heap to prioritize edges based on computed errors for collapsing.
        E_heap = []
        for i, e in enumerate(edges):
            # Compute new vertex position by averaging the positions of two vertices of the edge.
            v_0, v_1 = vs[e[0]], vs[e[1]]
            v_new = 0.5 * (v_0 + v_1)
            # Create a homogeneous coordinate for the new vertex position.
            v4_new = np.concatenate([v_new, np.array([1])])
            # Combine the quadric matrices of the two vertices.
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1
            # Compute the new error after potentially collapsing the edge.
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T))
            # Add the new error and edge index to the priority heap.
            heapq.heappush(E_heap, (E_new, i))
        
        # Mask to track which edges have been processed.
        mask = np.ones(self.edges_count, dtype=np.bool)
        # Vertex mask to keep track of active vertices.
        self.v_mask = np.ones(len(self.vs), dtype=np.bool)
        # Continuously collapse edges with the minimum error until reaching the target vertex count.
        while np.sum(self.v_mask) > target_v:
            # Extract the edge with the lowest error.
            E_0, edge_id = heapq.heappop(E_heap)
            edge = self.edges[edge_id]
            # Proceed with edge collapse if it's still marked as active.
            if mask[edge_id]:
                # Attempt to pool (collapse) the edge and update the mask.
                pool = self.pool_edge(edge_id, mask)
                if pool:
                    # Recompute errors for edges connected to the collapsed edge's vertices.
                    Q_0 = Q_s[edge[0]]
                    for vv_i in self.v2v[edge[0]]:
                        Q_1 = Q_s[vv_i]
                        Q_new = Q_0 + Q_1
                        v4_mid = np.concatenate([self.vs[edge[0]], np.array([1])])
                        E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T))
                        heapq.heappush(E_heap, (E_new, edge_id))
        # Final clean-up after modifications.
        self.clean(mask)
    
    def pool_main(self, target_v):
    # Load all necessary mesh data structures for operation.
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        # Initialize matrices for storing quadric errors and error scalars for each vertex.
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            # Fetch indices of faces associated with current vertex.
            f_s = np.array(list(vf[i]))
            # Extract face centers and normals corresponding to these faces.
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            # Compute the distance (d) term used in the quadric error matrices.
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            # Combine face normals and distance terms to form complete quadric data.
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            # Calculate the quadric error matrix for the current vertex.
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            # Prepare a homogeneous coordinate for the vertex.
            v4 = np.concatenate([v, np.array([1])])
            # Compute the initial error scalar for the current vertex.
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        # Initialize a heap to prioritize edges based on computed errors for collapsing.
        E_heap = []
        for i, e in enumerate(edges):
            # Compute new vertex position by averaging the positions of two vertices of the edge.
            v_0, v_1 = vs[e[0]], vs[e[1]]
            v_new = 0.5 * (v_0 + v_1)
            # Create a homogeneous coordinate for the new vertex position.
            v4_new = np.concatenate([v_new, np.array([1])])
            # Combine the quadric matrices of the two vertices.
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1
            # Compute the new error after potentially collapsing the edge.
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T))
            # Add the new error and edge index to the priority heap.
            heapq.heappush(E_heap, (E_new, i))
        
        # Mask to track which edges have been processed.
        mask = np.ones(self.edges_count, dtype=np.bool)
        # Vertex mask to keep track of active vertices.
        self.v_mask = np.ones(len(self.vs), dtype=np.bool)
        # Continuously collapse edges with the minimum error until reaching the target vertex count.
        while np.sum(self.v_mask) > target_v:
            # Extract the edge with the lowest error.
            E_0, edge_id = heapq.heappop(E_heap)
            edge = self.edges[edge_id]
            v_a = self.vs[edge[0]]
            v_b = self.vs[edge[1]]
            # Proceed with edge collapse if it's still marked as active.
            if mask[edge_id]:
                # Attempt to pool (collapse) the edge and update the mask.
                pool = self.pool_edge(edge_id, mask)
                if pool:
                    # Recompute errors for edges connected to the collapsed edge's vertices.
                    Q_0 = Q_s[edge[0]]
                    print(np.sum(self.v_mask), np.sum(mask))
                    for vv_i in self.v2v[edge[0]]:
                        Q_1 = Q_s[vv_i]
                        Q_new = Q_0 + Q_1
                        v4_mid = np.concatenate([self.vs[edge[0]], np.array([1])])
                        E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T))
                        heapq.heappush(E_heap, (E_new, edge_id))
        # Final clean-up after modifications.
        self.clean(mask)
    def pool_edge(self, edge_id, mask):
    # Vérifie si l'arête spécifiée contient des bords limites du maillage.
        if self.has_boundaries(edge_id):
            # Si l'arête est à la limite du maillage, ne pas l'effondrer.
            return False
        # Vérifie si l'effondrement de l'arête maintiendrait une topologie valide autour des sommets concernés.
        elif self.is_one_ring_valid(edge_id):
            # Si la topologie autour de l'arête est valide, procéder à l'effondrement de l'arête.
            self.merge_vertices(edge_id)
            # Marquer l'arête comme effondrée dans le masque.
            mask[edge_id] = False
            # Décrémenter le compteur d'arêtes actives du maillage.
            self.edges_count -= 1
            return True
        else:
            # Si l'effondrement de l'arête perturberait la topologie du maillage, ne pas l'effondrer.
            return False
    def merge_vertices(self, edge_id):
    # Supprimer l'arête spécifiée du maillage.
        self.remove_edge(edge_id)
        # Récupérer l'arête et les coordonnées des sommets concernés.
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # Calculer le point moyen entre les deux sommets de l'arête pour créer un nouveau sommet fusionné.
        self.vs[edge[0]] = 0.5 * (v_a + v_b)
        # Désactiver le second sommet de l'arête dans le masque de sommets, le marquant comme non actif.
        self.v_mask[edge[1]] = False
        # Mettre à jour toutes les références au second sommet pour qu'elles pointent vers le sommet fusionné.
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        # Réaffecter toutes les arêtes qui étaient connectées au second sommet pour qu'elles soient connectées au premier sommet.
        self.edges[mask] = edge[0]
    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            self.ve[v].remove(edge_id)
    
    def clean(self, edges_mask):
    # Applique le masque pour filtrer les arêtes, les arêtes GEMM et les côtés qui sont toujours actifs après l'effondrement.
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]

        # Préparer pour reconstruire la liste des arêtes pour chaque sommet.
        new_ve = []

        # Étendre le masque des arêtes pour inclure une valeur False à la fin, utile pour la gestion des indices.
        edges_mask = np.concatenate([edges_mask, [False]])
        # Initialiser un nouveau tableau d'indices qui pointera vers les nouveaux indices des arêtes.
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        # Définir le dernier élément à -1 pour gérer les références hors limites.
        new_indices[-1] = -1
        # Affecter les nouveaux indices aux positions vraies dans le masque.
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])

        # Mettre à jour les indices dans gemm_edges pour refléter les nouveaux indices des arêtes actives.
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]

        # Itérer sur chaque liste de ve (vertex edges) pour mettre à jour les indices selon les arêtes filtrées.
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # Si le sommet est toujours actif, mettre à jour ses références d'arêtes.
            # if self.v_mask[v_index]: (Commenté car il semble que cela devrait toujours être vérifié mais est omis ici.)
            for e in ve:
                # Utiliser le nouveau index si l'arête n'a pas été supprimée.
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)

        # Assigner la liste mise à jour des arêtes par sommet à self.ve.
        self.ve = new_ve
    def has_boundaries(self, edge_id):
    # Parcourir les arêtes connectées à l'arête spécifiée.
        for edge in self.gemm_edges[edge_id]:
            # Vérifier si l'arête ou ses voisins sont marqués par -1 (frontière).
            if edge == -1 or -1 in self.gemm_edges[edge]:
                print(edge_id, "is boundary")
                return True
        return False
    def is_one_ring_valid(self, edge_id):
    # Obtenir les sommets connectés à chaque sommet de l'arête.
        v_a = set(self.edges[self.ve[self.edges[edge_id, 0]]].reshape(-1))
        v_b = set(self.edges[self.ve[self.edges[edge_id, 1]]].reshape(-1))
        # Calculer l'intersection des ensembles, excluant les sommets de l'arête elle-même.
        shared = v_a.intersection(v_b).difference(set(self.edges[edge_id]))
        # Valider si exactement deux sommets sont partagés.
        return len(shared) == 2
    def __get_cycle(self, gemm, edge_id):
        cycles = []
        # Itérer pour chaque côté de l'arête.
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            # Suivre le cycle en suivant les connexions dans GEMM.
            for i in range(3):  # Note: cette boucle semble être un placeholder; la condition de sortie réelle peut varier.
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles
    
    def __cycle_to_face(self, cycle, v_indices):
        face = []
        # Parcourir les arêtes du cycle pour déterminer les sommets formant une face.
        for i in range(3):  # Assume des triangles pour simplification.
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face















