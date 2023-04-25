#Author: Kelby Amandy
#Date: 4/22/2023
#Tltle: Digit Recognizer

# Using this to create password recognizer
import tkinter as tk
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from PIL import Image, ImageOps, ImageGrab

# Load the saved digit recognizer model
model = load_model('digit_classifier.h5')

# Define a function to preprocess the canvas image and make a prediction
def recognize_digit():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab((x+2, y+2, x1-2, y1-2))

    # Convert the screenshot to grayscale, resize it to 28x28, and invert it
    img_pil = img.convert('RGB')
    img_pil = ImageOps.invert(ImageOps.grayscale(img_pil))
    img_pil = img_pil.resize((28, 28))

    # Convert the image to a numpy array and normalize the pixel values
    img = np.array(img_pil, dtype='float32') / 255.0

    # Showing what the grayscaled drawing looks like using matplotlib
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # Reshape the array to have a single channel and predict the digit using the model
    img = img.reshape(-1, 28, 28, 1)
    
    # predict the digit
    probs = model.predict(img)
    digit = probs.argmax()
    # Update the label with the predicted digit
    label.configure(text="The model guessed: " + str(digit) + ' with ' + str(np.max(probs)) + ' accuracy')

# Create the tkinter window
window = ctk.CTk()
window.geometry('550x550')
ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')
window.title("Digit Recognizer!")

# Create the canvas
canvas = ctk.CTkCanvas(window, width=400, height=400, bg='white')
canvas.place(relx=0.5, rely=0.45, anchor='center')

# Bind the left mouse button to draw on the canvas
def start_draw(event):
    canvas.create_oval(event.x-6.75, event.y-6.75, event.x+6.75, event.y+6.75, fill='black')
canvas.bind('<B1-Motion>', start_draw)

# Create the button to recognize the digit
button = ctk.CTkButton(window, text='Recognize', font=("Roboto", 14),command=recognize_digit)
button.place(relx=0.5, rely=0.9, anchor='center')

# Create a label to display the predicted digit
label = ctk.CTkLabel(window, text='Predicted digit: ', font=("Roboto", 14))
label.place(relx=0.5, rely=0.8, anchor='center')

# create a function to clear the canvas
def clear_canvas():
    canvas.delete('all')

# create a button for clearing the canvas
clear_button = ctk.CTkButton(window, text="Clear canvas", font=('Roboto', 14),command=clear_canvas)
clear_button.pack(pady=10)

# Start the tkinter main loop
window.mainloop()


