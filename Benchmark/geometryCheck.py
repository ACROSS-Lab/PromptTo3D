import trimesh
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, Entry, Label, Button, StringVar
import argparse
import time

def load_mesh(file_path):
    """
    Load a 3D mesh from a .obj file.
    """
    mesh = trimesh.load(file_path)
    return mesh

def find_connected_components(mesh):
    """
    Find connected components in the mesh.
    """
    components = mesh.split(only_watertight=False)
    return components

def filter_components(components, min_volume=0.00001, min_area=0.0001):
    """
    Filter out small components that might be holes or noise.
    """
    filtered_components = []
    for component in components:
        if component.volume > min_volume and component.area > min_area:
            filtered_components.append(component)
    return filtered_components

def check_discontinuities(components):
    """
    Check for geometric discontinuities by analyzing the number of connected components.
    """
    num_components = len(components)
    return num_components

def count_polygons(mesh):
    """
    Count the number of polygons (faces) in the mesh.
    """
    return len(mesh.faces)

def visualize_and_maybe_save_components(mesh, components, save_path, log_window):
    """
    Visualize the mesh and then each component by coloring them differently.
    Include the polygon count in the display.
    """
    polygon_count = count_polygons(mesh)
    scene = trimesh.Scene(mesh)
    update_log(log_window, f"Displaying the entire object with {polygon_count} polygons")
    scene.show()
    time.sleep(2)

    if len(components) > 1:
        for i, component in enumerate(components):
            polygon_count = count_polygons(component)
            scene = trimesh.Scene()
            component.visual.face_colors = trimesh.visual.random_color() if not hasattr(component.visual, 'face_colors') or component.visual.face_colors is None else component.visual.face_colors
            scene.add_geometry(component)
            update_log(log_window, f"Displaying component {i + 1} of {len(components)} with {polygon_count} polygons")
            update_log(log_window, f"The object is not geometrically GOOD with {len(components)} components")

            scene.show()
            time.sleep(2)

            if ask_save_component(i + 1):
                file_path = os.path.join(save_path, f"component_{i + 1}.obj")
                component.export(file_path)
                update_log(log_window, f"Component {i + 1} saved as {file_path} with {polygon_count} polygons")
    else:
        update_log(log_window, f"The object is  geometrically GOOD with {len(components)} components")
                

def ask_save_component(component_index):
    """
    Ask the user if they want to save the component using a graphical dialog.
    """
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Save Component", f"Do you want to save component {component_index}?")
    root.destroy()
    return result

def update_log(log_window, message):
    """
    Update the log window with a new message.
    """
    log_window.config(state=tk.NORMAL)
    log_window.insert(tk.END, message + "\n")
    log_window.config(state=tk.DISABLED)
    log_window.yview(tk.END)

def update_log_dimensions(log_window, width_var, height_var):
    """
    Update the log window dimensions based on user inputs.
    """
    try:
        new_width = int(width_var.get())
        new_height = int(height_var.get())
        log_window.config(width=new_width, height=new_height)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid integers for width and height.")

def main():
    parser = argparse.ArgumentParser(description='Process .obj files for mesh analysis.')
    parser.add_argument('image_path', help='Path to a single .obj file or a directory containing .obj files')
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Processing Log")
    
    # Variables for dimensions
    width_var = tk.StringVar(value='200')
    height_var = tk.StringVar(value='40')
    
    log_window = scrolledtext.ScrolledText(root, state=tk.DISABLED, width=150, height=50)
    log_window.pack()
    
    # Configuration controls for window size
    width_label = Label(root, text="Width:")
    width_label.pack(side=tk.LEFT)
    width_entry = Entry(root, textvariable=width_var)
    width_entry.pack(side=tk.LEFT)

    height_label = Label(root, text="Height:")
    height_label.pack(side=tk.LEFT)
    height_entry = Entry(root, textvariable=height_var)
    height_entry.pack(side=tk.LEFT)

    update_button = Button(root, text="Update Size", command=lambda: update_log_dimensions(log_window, width_var, height_var))
    update_button.pack(side=tk.LEFT)

    save_path = 'saved_components'
    os.makedirs(save_path, exist_ok=True)

    if os.path.isdir(args.image_path):
        for img in os.listdir(args.image_path):
            full_image_path = os.path.join(args.image_path, img)
            if os.path.isfile(full_image_path) and full_image_path.endswith('.obj'):
                update_log(log_window, f"Processing {full_image_path}...")
                mesh = load_mesh(full_image_path)
                components = find_connected_components(mesh)
                filtered_components = filter_components(components)
                num_components = check_discontinuities(filtered_components)
                update_log(log_window, f"Number of components in {os.path.basename(full_image_path)}: {num_components}")
                visualize_and_maybe_save_components(mesh, filtered_components, save_path, log_window)
    elif os.path.isfile(args.image_path) and args.image_path.endswith('.obj'):
        update_log(log_window, f"Processing {args.image_path}...")
        mesh = load_mesh(args.image_path)
        components = find_connected_components(mesh)
        filtered_components = filter_components(components)
        num_components = check_discontinuities(filtered_components)
        update_log(log_window, f"Number of components in {os.path.basename(args.image_path)}: {num_components}")
        visualize_and_maybe_save_components(mesh, filtered_components, save_path, log_window)
    else:
        update_log(log_window, f"The path '{args.image_path}' is not valid.")

    root.mainloop()

if __name__ == "__main__":
    main()
