import trimesh
import pyautogui
from vedo import load, Mesh
import vedo
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from os import path, getcwd
from glob import glob
import csv
import pyglet
"""
Ce script permet de visualiser les objets 3D générés par triposr et CRM, il faut que dans un dossier, vous ayez
tous vos objets rangés comme ceci :
./objet1, ./objet2, .... ./objetn (peu importe le noom du dossier ./objeti, il peut être renommé à votre guise il faut seulement que chaque asset 3D soit situé dans un dossier séparé)
ensuite chque dossier doit être organisé de la sorte, et contenir les 4 fichiers suivant nommés exactement de la sorte
* model1.obj : l'objet généré par le triposr
* model2.obj : la mesh de l'objet généré par le CRM
* model2.png : la couleur de l'objet généré par le CRM
"""
class ImageRater:
  def __init__(self):
        self.out_dir = None
        self.meshes = []
        self.prompts = []
        self.current_index = 0
        self.init_window()
        self.preference_var=None
        self.screen_width, self.screen_height = pyautogui.size()
        self.screen_width*=0.625
        self.screen_height*=0.625
        self.quarter_width = int(self.screen_width * 0.25)
        self.quarter_height = int(self.screen_height * 0.25)
  
  def get_out_dir(self):
      """
      get the directory where the 3D assets are stores, must be a relative path
      """
      self.out_dir = filedialog.askdirectory(title='Select Output Directory')
      if self.out_dir:
          self.window.destroy()
          self.get_meshes_and_prompts()

  def init_window(self):
      """
      initialisation of the Tkinter object
      """
      self.window = tk.Tk()
      self.window.title("Image to 3D model evaluation")
      out_dir_label = tk.Label(self.window, text="prompts and 3D assets repository")
      out_dir_label.pack()
      select_dir_button = tk.Button(self.window, text='Select your repository', command = self.get_out_dir)
      select_dir_button.pack()

  def load_meshes(self, filename):
    mesh = trimesh.load(filename+'/model1.obj')
    mesh2 = load(filename+'/model2.obj').texture(filename+'/model2.png')
    return mesh, mesh2


  def get_meshes_and_prompts(self):
      """
      Reads prompts from a single file and loads meshes from subfolders.
      """
      self.prompts = []
      self.meshes = []
      # Open the prompts file
      with open(path.join(self.out_dir, 'prompts.txt'), 'r') as f:
          self.prompts = [line.strip()[:] for line in f.readlines()]

      # Loop through subfolders and load meshes
      for folder in sorted(glob(path.join(self.out_dir, '[0-9]*/'))):  # Sort folders numerically
          cwd = getcwd()
          folder_name = path.relpath(folder, cwd)
          #this is for evaluation of two models, triposr and CRM
          if not path.isfile(path.join(folder, 'model1.obj')) or not path.isfile(path.join(folder, 'model2.obj')) or not path.isfile(path.join(folder, 'model2.png')):
              print(f"Warning: The folder {folder_name} - Isn't complete, it needs to have 2 .obj files and one .png file, please fix it this folder is skipped")
              continue
          # Load meshes using your existing function
          trimesh_mesh, vedo_mesh = self.load_meshes(folder_name)
          self.meshes.append((trimesh_mesh, vedo_mesh))

      # Validate number of meshes and prompts
      if len(self.meshes) != len(self.prompts):
          raise ValueError("Error: Number of folders must match the number of prompts.")
      self.create_rating_window()
      self.show_image_and_prompt()

  def save_and_next(self):
    """
    Saves the preference (Mesh 1 or Mesh 2) for the current prompt and displays the next one (if available).
    """
    selected_preference = self.preference_var.get()
    if selected_preference == "Select Preference":
        print("Please select a preference (Mesh 1 or Mesh 2) before saving.")
        return
    
    self.current_index += 1
    with open(self.out_dir+"/scores.csv", "a", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow([self.current_index, selected_preference, self.prompts[self.current_index-1]])

    if self.current_index < len(self.meshes):
        self.show_image_and_prompt()
    else:
        self.show_completion_message()
        self.quit_button.config(text="Close", command=self.quit_app)

    # Reset preference selection
    self.preference_var.set("Select Preference")
  
  def show_completion_message(self):
    """
    Displays a message indicating that all 3D assets have been rated.
    """
    message = "All 3D assets have been rated!"
    completion_label = tk.Label(self.rating_window, text=message)
    completion_label.pack()

  def quit_app(self):
    """
    Quits the application and destroys the rating window.
    """
    self.rating_window.destroy()


  def create_rating_window(self):
    """
    Creates a new window for displaying the mesh, prompt, and rating options.
    """
    self.rating_window = Tk()
    self.rating_window.title("Rate Meshes")
    self.prompt_label = tk.Label(self.rating_window)
    self.prompt_label.pack()
    self.preference_var = tk.StringVar(self.rating_window)
    self.preference_var.set("Select Preference")  # Initial value
    preference_options = [("Mesh 1"), ("Mesh 2")]
    preference_combobox = ttk.Combobox(self.rating_window, textvariable=self.preference_var, values=preference_options)
    preference_combobox.pack()

    # Button to save and go to next
    self.save_and_next_button = tk.Button(self.rating_window, text="Save & Next", command=self.save_and_next)
    self.save_and_next_button.pack()

    # Button to quit
    self.quit_button = tk.Button(self.rating_window, text="Quit", command=self.quit_app)
    self.quit_button.pack()

  def show_trimesh_mesh(self, mesh):
      scene = trimesh.Scene([mesh])
      from trimesh.viewer import windowed
      viewer = windowed.SceneViewer(scene, resolution =(self.quarter_width*3, self.quarter_height*3))
      viewer.set_location(1000, 1000)

  def show_vedo_mesh(self, mesh):
      """
      Creates a Tkinter window to display the vedo mesh using vedo.show().
      """
      pos = (0, int(self.screen_height - self.quarter_height))  # Top-left corner at (0, remaining height)
      size = (self.quarter_width*3, self.quarter_height*3)
      vedo_window = mesh.show(pos = pos, size=size).close()
      vedo_window.close_on_empty_queue = True  # Close vedo window when Tkinter closes


  def show_image_and_prompt(self):
      """
      Displays the prompts and opens separate windows for each mesh.
      """
      self.prompt_label.config(text=self.prompts[self.current_index])
      trimesh_mesh1, vedo_mesh2 = self.meshes[self.current_index]
      self.show_vedo_mesh(vedo_mesh2)
      self.show_trimesh_mesh(trimesh_mesh1)

if __name__ == "__main__":
    rater = ImageRater()
    tk.mainloop()