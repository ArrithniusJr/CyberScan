from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import ttk
import customtkinter
from CTkMessagebox import CTkMessagebox

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("Welcome | CyberScan - Content Detector")
root.geometry("900x600")
root.resizable(False, False)

canvas_login = customtkinter.CTkFrame(master=root)
canvas_login = customtkinter.CTkFrame(root, width=900, height=600)
canvas_login.pack(pady=20, padx=60, fill="both", expand=True)

# ---------- Logo ---------# 
light_image = Image.open("logo-no-background.png").convert("RGBA")
dark_image = Image.open("logo-no.png").convert("RGBA")

logo_image = customtkinter.CTkImage(light_image=light_image, dark_image=dark_image, size=(310, 150))
logo_label = customtkinter.CTkLabel(canvas_login, image=logo_image)
logo_label.configure(text="")
logo_label.place(relx=0.5, rely=0.5, anchor="center")
logo_label.pack(pady=20)

# ---------- Main Container ---------# 
container = customtkinter.CTkFrame(master=canvas_login)
container = customtkinter.CTkFrame(canvas_login, width=300, height=300)
container.pack(pady=20, padx=250, fill="both", expand=True)

# ---------- Username ---------# 
def user_onfocus(e):
    username.delete(0, 'end')

def user_offfocus(e):
    name=username.get() 
    if name=='':
        username.insert(0, 'Enter your email')
        username.configure(fg='gray')

username_label = customtkinter.CTkLabel(container, text="Username:")
username_label.pack()
username = customtkinter.CTkEntry(master=container, placeholder_text="Enter your email")
username.bind('<FocusIn>', user_onfocus) 
username.bind('<FocusOut>', user_offfocus) 
username.pack(pady=12, padx=10)
username.configure(height=35, width=210)

# ---------- Password---------# 
def pass_onfocus(e):
    password.delete(0, 'end')

def pass_offfocus(e):
    name=password.get() 
    if name=='':
        password.insert(0, 'Enter your password')
        password.configure(fg='gray')

password_label = customtkinter.CTkLabel(container, text="Password:")
password_label.pack()
password = customtkinter.CTkEntry(master=container, placeholder_text="Enter your password", show="*")
password.bind('<FocusIn>', pass_onfocus) 
password.bind('<FocusOut>', pass_offfocus)
password.pack(pady=12, padx=10)
password.configure(height=35, width=210)

def login(canvas_to_destroy):
    user_val = username.get()
    pass_val = password.get()
    if user_val == "" and pass_val == "":
        print("Login works")
        
    else:
        msg = CTkMessagebox(title="Login Error!", message="Invalid username and password",
                icon="cancel", option_1="Exit", option_2="Retry")
        
        if msg.get()=="Exit":
            root.destroy()


button = customtkinter.CTkButton(master=container, text="Login", command=lambda: login(canvas_login), cursor='hand2')
button.pack(pady=12, padx=10)

account = customtkinter.CTkLabel(master=container, text="Dont't have an account?")
account.pack_forget()

register = customtkinter.CTkLabel(master=container, text="Register now", cursor='hand2')
register.pack_forget()

# Put the labels on the same line
account_register_labels = customtkinter.CTkLabel(master=container, text="", anchor="center")
account_register_labels.pack(pady=12, padx=10)
account_register_labels.configure(compound="center", text=f"{account.cget('text')} {register.cget('text')}")

root.bind('<Return>', lambda event: login(canvas_login))

root.mainloop()