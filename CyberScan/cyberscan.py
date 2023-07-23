from tkinter import *
from PIL import ImageTk, Image

class RoundedText(Text):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(borderwidth=0, highlightthickness=0, bg="#f2f2f2", fg="#000")
        self.bind("<FocusIn>", self.on_focus_in)
        self.bind("<FocusOut>", self.on_focus_out)
        
        self.hint_text = "Enter your content here"
        self.insert("1.0", self.hint_text)
        self.tag_configure("hint", foreground="#777")

    def on_focus_in(self, event):
        if self.get("1.0", "end-1c") == self.hint_text:
            self.delete("1.0", "end-1c")
            self.tag_configure("hint", foreground="#000")
        self.config(bg="#e2e2e2")

    def on_focus_out(self, event):
        if not self.get("1.0", "end-1c"):
            self.insert("1.0", self.hint_text, "hint")
        self.config(bg="#f2f2f2")


def clear_text():
    input_text.delete(1.0, END)
    result_container.config(text="")

def analyze_content():
    input_text_value = input_text.get(1.0, END).strip()
    result_container.config(text="Detected content: " + input_text_value)

root = Tk()
root.title("CyberScan - Content Detector")
root.attributes("-zoomed", True)  # Maximize the window

# Set background image
background_image = ImageTk.PhotoImage(file="background_image.jpg")
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

logo_image = PhotoImage(file="logo-no.png").subsample(4)
logo_label = Label(root, image=logo_image)
logo_label.place(relx=0.5, rely=0.1, anchor="center")

container = Frame(root, padx=20, pady=20, highlightthickness=0)
container.place(relx=0.5, rely=0.5, anchor="center")

input_text = RoundedText(container, height=3, width=40, bd=0, highlightbackground="#f2f2f2", highlightthickness=1)
input_text.pack(pady=10)

button_container = Frame(container)
button_container.pack()

clear_button = Button(button_container, text="Clear", bg="#ff0000", fg="#fff", padx=15, pady=5, bd=0, command=clear_text)
clear_button.pack(side=LEFT)

# Add a horizontal space
space_label = Label(button_container, width=1)
space_label.pack(side=LEFT)

analyze_button = Button(button_container, text="Analyze", bg="#008000", fg="#fff", padx=15, pady=5, bd=0, command=analyze_content)
analyze_button.pack(side=LEFT)

# Add a vertical space
space_label = Label(button_container, height=4)
space_label.pack(side=LEFT)

result_container = Label(container, text="", font=("Arial", 12), wraplength=400)
result_container.pack()

root.mainloop()

