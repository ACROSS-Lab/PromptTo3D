import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
import csv
import PIL








class ImageRater:
    """
    class that will help to evaluate the prompts generated by stablediffusion
    """
    def __init__(self):
        self.out_dir = None
        self.images = []
        self.prompts = []
        self.current_index = 0
        self.init_window()

    def init_window(self):
        """
        initialisation of the Tkinter object
        """
        self.window = tk.Tk()
        self.window.title("Stable Diffusion evaluation")

        out_dir_label = Label(self.window, text="Images and prompts repository")
        out_dir_label.pack()
        self.out_dir_entry = Entry(self.window)
        self.out_dir_entry.pack()

        select_dir_button = Button(self.window, text='Select this repository', command = self.get_out_dir)
        delect_dir_button.pack()

        def get_out_dir(self):
            """
            get the directory where the images are stores, must be a relative path
            """
            self.out_dir = filedialog.askdirectory(title='Select Output Directory')
            if self.out_dir:
                self.window.destroy()
                self.process_images()

    def read_data(self):
        """
        application qui lit les images et les prompts,
        attention, out_dir doit être un chemin relatif
        pas un chemin absolu. out_dir fdoit être le fichier 
        contenant les .png et le .txt
        """
        for img in glob.glob(self.out_dir+'*.png'):
            self.images.append(PIL.Image.open(img))

        for files in glob.glob(self.out_dir+'*.txt'):
            with open(files, 'r') as f:
                self.prompts = f.readlines()

        for i in range (len(prompts)-1):
            self.prompts[i]=prompts[i][:-2]

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

        # Display image
        self.image_label = Label(self.rating_window)
        self.image_label.pack()

        # Display prompt
        self.prompt_label = Label(self.rating_window)
        self.prompt_label.pack()

        # Entry for score
        self.score_entry = Entry(self.rating_window)
        self.score_entry.pack()

        # Button to save score and go to next image
        self.save_and_next_button = Button(self.rating_window, text="Save & Next", command=self.save_and_next)
        self.save_and_next_button.pack()

        # Button to quit the application
        self.quit_button = Button(self.rating_window, text="Quit", command=self.quit_app)
        self.quit_button.pack()

    def show_image_and_prompt(self):
        """
        Displays the current image and prompt in the rating window.
        """
        image_path = self.images[self.current_index]
        prompt = self.prompts[self.current_index]
        self.image_label.config(image=Image.open(image_path))
        self.prompt_label.config(text=prompt)

    def save_and_next(self):
        """
        Saves the score for the current image-prompt pair and displays the next one (if available).
        """
        score = self.score_entry.get()
        # Implement logic to save score and image/prompt data to CSV (e.g., using csv library)
        # ...

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
    rater.init_window()
    tk.mainloop()


    