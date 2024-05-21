import trimesh
from vedo import load
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, filedialog
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import glob
import csv

class ImageRater:
  def __init__(self):
        self.out_dir = None
        self.meshes = []
        self.prompts = []
        self.current_index = 0
        self.photo_image = None
        self.init_window()
        self.preference_var=None
  
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
    #mesh.show()
    mesh2 = load(filename+'/model2.obj').texture(filename+'/model2.png')
    #mesh2.show()  
    return mesh, mesh2


  def get_meshes_and_prompts(self):
      """
      Reads prompts from a single file and loads meshes from subfolders.
      """
      self.prompts = []
      self.meshes = []

      # Open the prompts file
      with open(os.path.join(self.out_dir, 'prompts.txt'), 'r') as f:
          self.prompts = [line.strip()[:-2] for line in f.readlines()]

      # Loop through subfolders and load meshes
      for folder in sorted(glob.glob(os.path.join(self.out_dir, '[0-9]*/'))):  # Sort folders numerically
          cwd = os.getcwd()
          folder_name = os.path.relpath(folder, cwd)
          #this is for evaluation of two models, triposr and CRM
          if not os.path.isfile(os.path.join(folder, 'model1.obj')) or not os.path.isfile(os.path.join(folder, 'model2.obj')) or not os.path.isfile(os.path.join(folder, 'model2.png')):
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

    # Validate selection


    if selected_preference == "Select Preference":
        print("Please select a preference (Mesh 1 or Mesh 2) before saving.")
        return
    

    self.current_index += 1

    with open(self.out_dir+"/scores.csv", "a", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow([self.current_index, selected_preference, self.prompts[self.current_index]])


    if self.current_index < len(self.meshes):
        self.show_image_and_prompt()
    else:
        self.show_completion_message()
        self.quit_button.config(text="Close", command=self.quit_app)

    # Reset preference selection
    self.preference_var.set("Select Preference")
  
  def show_completion_message(self):
    """
    Displays a message indicating that all images have been rated.
    """
    message = "All images have been rated!"
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

    # Get window width and height
    window_width = self.rating_window.winfo_screenwidth()
    window_height = self.rating_window.winfo_screenheight()

    self.prompt_label = tk.Label(self.rating_window)
    self.prompt_label.pack()

    # Image label
    self.image_label = tk.Label(self.rating_window)
    self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



    # Preference selection (combobox)
    self.preference_var = tk.StringVar(self.rating_window)
    self.preference_var.set("Select Preference")  # Initial value
    preference_options = [("Mesh 1",1), ("Mesh 2",2)]
    preference_combobox = ttk.Combobox(self.rating_window, textvariable=self.preference_var, values=preference_options)
    preference_combobox.pack()

    # Button to save and go to next
    self.save_and_next_button = tk.Button(self.rating_window, text="Save & Next", command=self.save_and_next)
    self.save_and_next_button.pack()

    # Button to quit
    self.quit_button = tk.Button(self.rating_window, text="Quit", command=self.quit_app)
    self.quit_button.pack()

  def show_image_and_prompt(self):
    """
    Displays the prompts and both meshes using mayavi.
    """
    trimesh_mesh1, trimesh_mesh2 = self.meshes[self.current_index]

    # Update the prompt label
    self.prompt_label.config(text=self.prompts[self.current_index])
    
    trimesh_mesh1.show()
    trimesh_mesh2.show().close()


if __name__ == "__main__":
    rater = ImageRater()
    #rater.init_window()
    tk.mainloop()