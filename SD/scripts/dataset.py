import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
import csv
from PIL import Image, ImageTk
import json
import requests
from PIL import Image
import argparse

class ImageRater:
    """
    class that will help to evaluate the prompts generated by stablediffusion
    """
    def __init__(self, start_index):
        self.out_dir = None
        self.images = []
        self.names = []
        self.descriptions = []
        self.current_index = 0
        self.photo_image_temp=None
        self.prompt_temp=None
        self.init_window()
        self.score_1_var = None
        self.max_height = 0
        self.max_width = 0
        self.data = []
        self.ids = []
        self.image = None
        self.name = None
        self.description = None
        self.start_index = start_index

    def get_out_dir(self):
        """
        get the directory where the json is stored, must be a relative path
        """
        self.out_dir = filedialog.askdirectory(title='Select Output Directory')
        if self.out_dir:
            self.window.destroy()
            self.read_data()

    def init_window(self):
        """
        initialisation of the Tkinter object
        """
        self.window = tk.Tk()
        self.window.title("Stable Diffusion evaluation")
        out_dir_label = tk.Label(self.window, text="Images and prompts repository")
        out_dir_label.pack()
        select_dir_button = tk.Button(self.window, text='Select your repository', command = self.get_out_dir)
        select_dir_button.pack()

    def read_data(self):
        """
        application qui lit les images et les prompts,
        attention, out_dir doit être un chemin relatif
        pas un chemin absolu. out_dir doit être le fichier
        contenant les .png et le .txt
        """
        with open(self.out_dir +"/000-023.json", "r") as read_file:
            self.data = json.load(read_file)
        self.ids = list(self.data.keys())[self.start_index:]
        self.images = [self.data[i]['thumbnails']['images'][0]['url'] for i in self.ids][self.start_index:]
        self.names =[self.data[i]['name'] for i in self.ids][self.start_index:]
        #just take the first line of each description
        self.descriptions = [self.data[i]['description'].partition('\n')[0] for i in self.ids][self.start_index:]
        self.create_rating_window()
        self.show_image_and_prompt()

    def create_rating_window(self):
        """
        Creates a new window for displaying the image, prompt, entry for score, and buttons.
        """
        self.rating_window = Tk()
        self.rating_window.title("Rate Image")

        # Get window width and height
        window_width = self.rating_window.winfo_screenwidth()
        window_height = self.rating_window.winfo_screenheight()

        # Set maximum image dimensions (adjust as needed)
        self.max_width = int(window_width * 0.8)  # 80% of screen width
        self.max_height = int(window_height * 0.6)  # 60% of screen height

        score_1_frame = tk.Frame(self.rating_window)
        score_1_label = tk.Label(score_1_frame, text="correspondance image-nom/description")
        score_1_label.pack(side="left")
        self.score_1_var = tk.IntVar()
        score_1_button1 = tk.Radiobutton(score_1_frame, text="None", variable=self.score_1_var, value=0)
        score_1_button2 = tk.Radiobutton(score_1_frame, text="Name", variable=self.score_1_var, value=1)
        score_1_button3 = tk.Radiobutton(score_1_frame, text="Description", variable=self.score_1_var, value=2)
        score_1_button1.pack(side="left")
        score_1_button2.pack(side="left")
        score_1_button3.pack(side="left")
        score_1_frame.pack()

        # Button to save score and go to next image
        self.save_and_next_button = tk.Button(self.rating_window, text="Save & Next", command=self.save_and_next)
        self.save_and_next_button.pack()

        # Button to quit the application
        self.quit_button = tk.Button(self.rating_window, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        # Display prompt
        self.name_label = tk.Label(self.rating_window)
        self.name_label.pack()

        # Display prompt
        self.description_label = tk.Label(self.rating_window)
        self.description_label.pack()

        self.index_label = tk.Label(self.rating_window)
        self.index_label.pack()

        # Display image
        self.image_label = tk.Label(self.rating_window)
        self.image_label.pack(side = tk.BOTTOM, fill="both", expand = True, anchor = 's')
        self.show_image_and_prompt()

    def show_image_and_prompt(self):
        """
        Displays the current image and prompt in the rating window.
        """
        self.image = self.images[self.current_index]
        self.name = self.names[self.current_index]
        self.description = self.descriptions[self.current_index]
        #le try permet d'éviter les problèmes d'url non foctionnels
        try :
            image_PIL = Image.open(requests.get(self.image, stream=True).raw)
        except  Exception as e:
            # Handle cases where the request fails or the image format is unsupported
            print(f"Error processing image {url}: {e}")
            self.show_url_error_message()
        # Resize image while maintaining aspect ratio
        image_width, image_height = image_PIL.size
        if image_width > self.max_width or image_height > self.max_height:
            # Calculate resize ratio
            scale_ratio = min(self.max_width / image_width, self.max_height / image_height)
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            image_PIL = image_PIL.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)  # Resize with antialiasing
        self.photo_image_temp = ImageTk.PhotoImage(image_PIL)
        self.image_label.config(image=self.photo_image_temp)
        self.name_label.config(text="Name= " + self.name)
        self.description_label.config(text="Description = " + self.description)
        self.index_label.config(text="Indice : " + str(self.current_index + self.start_index))

    def save_and_next(self):
        """
        Saves the score for the current image-prompt pair and displays the next one (if available).
        """
        score1 = self.score_1_var.get()
        if score1 > 0 :
            if score1 == 1 :
                score1 = self.name
            else :
                score1 = self.description
            # Save score to CSV
            image_name = self.images[self.current_index]
            with open(self.out_dir+"/dataset_finetune.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_name, score1, self.current_index, self.ids[self.current_index]])  # Save image name and adequate_text
        self.current_index += 1
        if self.current_index < len(self.images):
            self.show_image_and_prompt()
        else:
            self.show_completion_message()
            self.quit_button.config(text="Close", command=self.quit_app)
        self.score_1_var.set(0)

    def show_completion_message(self):
        """
        Displays a message indicating that all images have been rated.
        """
        message = "All images have been rated!"
        completion_label = tk.Label(self.rating_window, text=message)
        completion_label.pack()

    def show_url_error_message(self):
        """
        Displays a message indicating that all images have been rated.
        """
        message = "Il y a un problème avec cet URL, veuillez le passer"
        url_label = tk.Label(self.rating_window, text=message)
        url_label.pack()

    def quit_app(self):
        """
        Quits the application and destroys the rating window.
        """
        self.rating_window.destroy()

# Run the main loop
def main(start_index):
    rater = ImageRater(start_index)
    #rater.init_window()
    tk.mainloop()

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description = 'Faire un dataset d\'objet 3D')
    parser.add_argument(
        "--start_index",
        help = "Indice de départ de l'évaluation",
        default = 0)
    args = parser.parse_args()
    main(**vars(args))