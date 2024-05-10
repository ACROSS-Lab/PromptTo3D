import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
import csv
import PIL
from PIL import Image, ImageTk
import glob
import os


class ImageRater:
    """
    class that will help to evaluate the prompts generated by stablediffusion
    """
    def __init__(self):
        self.out_dir = None
        self.images = []
        self.prompts = []
        self.current_index = 0
        self.photo_image_temp=None
        self.prompt_temp=None
        self.init_window()




    def get_out_dir(self):
        """
        get the directory where the images are stores, must be a relative path
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
        pas un chemin absolu. out_dir fdoit être le fichier
        contenant les .png et le .txt
        """
        for img in glob.glob(os.path.relpath(self.out_dir+'/*.png')):
            self.images.append(PIL.Image.open(img))




        for files in glob.glob(os.path.relpath(self.out_dir+'/*.txt')):
            with open(files, 'r') as f:
                self.prompts = f.readlines()




        for i in range (len(self.prompts)-1):
            self.prompts[i]=self.prompts[i][:-2]




        if len(self.images) != len(self.prompts):
            raise ValueError(f"Error, There must be as many prompts as images, here we found {len(self.images)} images and {len(self.prompts)} prompts...")
        self.create_rating_window()
        self.show_image_and_prompt()






       




    def create_rating_window(self):
        """
        Creates a new window for displaying the image, prompt, entry for score, and buttons.
        """
        self.rating_window = Tk()
        self.rating_window.title("Rate Image")










        # Display prompt
        self.prompt_label = tk.Label(self.rating_window)
        self.prompt_label.pack()




        # Entry for score
        self.score_entry = tk.Entry(self.rating_window)
        self.score_entry.pack()




        # Button to save score and go to next image
        self.save_and_next_button = tk.Button(self.rating_window, text="Save & Next", command=self.save_and_next)
        self.save_and_next_button.pack()




        # Button to quit the application
        self.quit_button = tk.Button(self.rating_window, text="Quit", command=self.quit_app)
        self.quit_button.pack()


        # Display image
        self.image_label = tk.Label(self.rating_window)
        self.image_label.pack(fill="both", expand = True)


        self.show_image_and_prompt()




    def show_image_and_prompt(self):
        """
        Displays the current image and prompt in the rating window.
        """
        image = self.images[self.current_index]
        self.prompt_temp = self.prompts[self.current_index]
        self.photo_image_temp = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo_image_temp)
        self.prompt_label.config(text=self.prompt_temp)




    def save_and_next(self):
        """
        Saves the score for the current image-prompt pair and displays the next one (if available).
        """
        score = self.score_entry.get()


        # Save score to CSV
        image_name = os.path.basename(self.images[self.current_index].filename)  # Get image filename
        with open(self.out_dir+"/scores.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, score, self.prompts[self.current_index]])  # Save image name, score, and prompt


        # Clear score entry for next image
        self.score_entry.delete(0, tk.END)


        self.current_index += 1
        if self.current_index < len(self.images):
            self.show_image_and_prompt()
        else:
            self.show_completion_message()
            self.quit_button.config(text="Close", command=self.quit_app)






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




# Run the main loop
if __name__ == "__main__":
    rater = ImageRater()
    tk.mainloop()


