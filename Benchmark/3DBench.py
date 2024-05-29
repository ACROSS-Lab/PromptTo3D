import trimesh


def screenshot_the_mesh(mesh, prompt):
    scene = mesh.scene()
    viewpoints = [] #list of the viewpoints 
    for i, viewpoint in enumerate(viewpoints):
        # Set camera pose
        camera_old, _geometry = scene.graph[scene.camera.name]
        scene.graph[scene.camera.name] = np.dot(viewpoint, camera_old)
        # Save image
        filename = filename_format.format(i)
        png = scene.save_image(resolution=resolution, visible=True)
        with open(filename, "wb") as f:
            f.write(png)
        # Reset camera pose (optional)
        scene.graph[scene.camera.name] = camera_old