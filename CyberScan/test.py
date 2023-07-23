from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import ttk
import customtkinter
from CTkMessagebox import CTkMessagebox

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("CyberScan | Content Detector System")
root.geometry("900x600")
root.resizable(False, False)

canvas_login = None
canvas_dashboard = None

###--------- Sign In Function -------###
def setup_login_page():
    global canvas_login

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
            username.insert(0,'Enter your email')

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

    password_label = customtkinter.CTkLabel(container, text="Password:")
    password_label.pack()
    password = customtkinter.CTkEntry(master=container, placeholder_text="Enter your password", show="*")
    password.bind('<FocusIn>', pass_onfocus) 
    password.bind('<FocusOut>', pass_offfocus)
    password.pack(pady=12, padx=10)
    password.configure(height=35, width=210)

    def login():
        user_val = username.get()
        pass_val = password.get()
        if user_val == "admin" and pass_val == "12345":
            switch_to_dashboard()
        else:
            msg = CTkMessagebox(title="Login Error!", message="Invalid username and password",
                    icon="cancel", option_1="Exit", option_2="Re-enter")
            if msg.get()=="Exit":
                root.destroy()

    button = customtkinter.CTkButton(master=container, text="Login", command=login, cursor='hand2')
    button.pack(pady=12, padx=10)

    account = customtkinter.CTkLabel(master=container, text="Dont't have an account?")
    account.pack_forget()

    register = customtkinter.CTkLabel(master=container, text="Register now", cursor='hand2')
    register.pack_forget()

    # Put the labels on the same line
    account_register_labels = customtkinter.CTkLabel(master=container, text="", anchor="center")
    account_register_labels.pack(pady=12, padx=10)
    account_register_labels.configure(compound="center", text=f"{account.cget('text')} {register.cget('text')}")

    root.bind('<Return>', lambda event: login())

########################################
########################################
### ------- Dashboard Function ----- ### 
########################################
########################################
def switch_to_dashboard():
    global canvas_dashboard

    # ---------- COLUMN 1 ---------# 
    # ---------- Account Frame ---------# 
    account_frame = customtkinter.CTkFrame(master=root)
    account_frame = customtkinter.CTkFrame(root, width=200, height=600)
    account_frame.pack(pady=20, padx=(20, 15), fill="both", expand=True, side="left")

    text = "Account"
    description_label = customtkinter.CTkLabel(account_frame, text=text, wraplength=190, font=("Liberation Sans", 16, "bold"), anchor="w")
    description_label.pack(pady=15, padx=10, fill="x", expand=False)

    # ---------- Username Parameter ---------# 
    text = "Username"
    username_name = customtkinter.CTkLabel(account_frame, text=text, wraplength=190, anchor="w")
    username_name.pack(padx=10, fill="x")
    username_name.place(y=40, x=10)

    # ---------- Email Parameter ---------# 
    text = "Email"
    username_name = customtkinter.CTkLabel(account_frame, text=text, wraplength=190, anchor="w")
    username_name.pack(padx=10, fill="x")
    username_name.place(y=68, x=10)

    # ---------- Log out button ---------# 
    def logout():
        canvas_dashboard.destroy()

    log_out = customtkinter.CTkButton(master=account_frame, text="Log out", command=logout, cursor='hand2')
    log_out.pack(pady=53, padx=10, fill="x")
    log_out.configure(height= 40)

    # ---------- COLUMN 2 --------- # 
    # ---------- Dashboard Frame ---------# 
    canvas_dashboard = customtkinter.CTkFrame(master=root)
    canvas_dashboard = customtkinter.CTkFrame(root, width=600, height=600)
    canvas_dashboard.pack(pady=20, padx=(10, 20), fill="both", expand=True, side="left")

    # ---------- Logo ---------# 
    light_image = Image.open("logo-no-background.png").convert("RGBA")
    dark_image = Image.open("logo-no.png").convert("RGBA")

    logo_image = customtkinter.CTkImage(light_image=light_image, dark_image=dark_image, size=(310, 150))
    logo_label = customtkinter.CTkLabel(canvas_dashboard, image=logo_image)
    logo_label.configure(text="")
    logo_label.place(relx=0.5, rely=0.5, anchor="center")
    logo_label.pack(pady=20)

    # ---------- Container Frame ---------# 
    container = customtkinter.CTkFrame(master=canvas_dashboard)
    container = customtkinter.CTkFrame(canvas_dashboard, width=300, height=250)
    container.pack(pady=0, padx=100, fill="both", expand=True)

    # ---------- Description ---------# 
    text = "A CyberScan software is a system that lets you test your content to find out if it is harmful or not. Use the below ChatBox to analyse your message before sending."
    description_label = customtkinter.CTkLabel(container, text=text, wraplength=390)
    description_label.pack(pady="15")

    #---------- Separator ---------
    style = ttk.Style()
    style.configure("Horizontal.TSeparator", background="black") 
    line_separator = ttk.Separator(container, orient="horizontal", style="Horizontal.TSeparator")
    line_separator.pack(fill="x", padx=50, pady=10)

    # ---------- Message ---------# 
    def msg_onfocus(e):
        message.delete(0, 'end')

    def msg_offfocus(e):
        name=message.get() 
        if name=='':
            message.insert(0, 'Enter your content here')

    message_txt = customtkinter.CTkLabel(container, text="Message:")
    message_txt.pack()
    message = customtkinter.CTkEntry(master=container, placeholder_text="Enter your content here")
    message.bind('<FocusIn>', msg_onfocus) 
    message.bind('<FocusOut>', msg_offfocus) 
    message.pack(pady=2, padx=10)
    message.configure(height=40, width=300)

    # ---------- Analyze button ---------# 
    def analyze_content():
        input_text_value = message.get().strip()
        result_container.configure(text="Results: " + input_text_value)

    analyse_button = customtkinter.CTkButton(master=container, text="Analyse", command=analyze_content, cursor='hand2')
    analyse_button.pack(side="left", padx=(130, 10), anchor="center")
    analyse_button.configure(height=40)

    # ---------- Clear button ---------# 
    clear_icon = customtkinter.CTkImage(Image.open("bin.png").convert("RGBA"))

    def clear_text():
        message.delete(0, 'end')
        result_container.configure(text="Results: ")

    clear_button = customtkinter.CTkButton(master=container, image=clear_icon, command=clear_text, cursor='hand2', fg_color="#990000", hover_color="#660000")
    clear_button.configure(height=40, width=15, text="")
    clear_button.pack(side="left", padx=(5, 100), anchor="center")

    # ---------- Results Frame ---------# 
    container2 = customtkinter.CTkFrame(master=canvas_dashboard)
    container2 = customtkinter.CTkFrame(canvas_dashboard, width=100, height=100)
    container2.pack(pady=15, padx=180, fill="both", expand=False)

    # ---------- Result Container ---------# 
    result_container = customtkinter.CTkLabel(container2, text="Results: ", wraplength=390)
    result_container.pack(pady=15)

    canvas_login.destroy()

    root.bind('<Return>', lambda event: analyze_content())

    root.mainloop()

# --------- INITIALISING -------- #
setup_login_page()

root.mainloop()