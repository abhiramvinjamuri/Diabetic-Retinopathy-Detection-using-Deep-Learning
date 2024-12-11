import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import HybridResNetDenseNetModel

# User data file
user_data_file = "user_data.json"

# Load model
model = HybridResNetDenseNetModel(num_classes=7)
model.load_state_dict(torch.load("diabetic_retinopathy_model.pth", map_location=torch.device('cpu')), strict=False)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# User data handling functions
def load_user_data():
    try:
        with open(user_data_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_user_data(users):
    with open(user_data_file, 'w') as file:
        json.dump(users, file)

def create_user(username, password):
    users = load_user_data()
    if username in users:
        messagebox.showwarning("Warning", "User already exists!")
        return
    users[username] = password
    save_user_data(users)
    messagebox.showinfo("Success", "User created successfully!")

def validate_user(username, password):
    users = load_user_data()
    return username in users and users[username] == password

def signup():
    username = username_entry.get()
    password = password_entry.get()
    if username and password:
        create_user(username, password)
    else:
        messagebox.showwarning("Warning", "Please enter both username and password.")

def login():
    username = username_entry.get()
    password = password_entry.get()
    if validate_user(username, password):
        messagebox.showinfo("Success", "Login successful!")
        main_window()
    else:
        messagebox.showwarning("Failed", "Invalid username or password.")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction, confidence_score, severity_index = predict_image(file_path)
        result_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence_score:.2f}%\nSeverity Index: {severity_index}/7")
        update_severity_scale(severity_index)
        plot_confidence_graph()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

    severity = predicted.item()
    labels = {
        0: "No DR signs",
        1: "Mild (or early) NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Very Severe NPDR",
        5: "PDR",
        6: "Advanced PDR"
    }

    severity_label = labels[severity]
    confidence_score = confidence * 100  # Convert to percentage
    severity_index = severity + 1  # Severity index is 1-based
    return severity_label, confidence_score, severity_index

def update_severity_scale(severity_index):
    max_severity = 7  # The maximum severity level
    severity_scale = severity_index / max_severity * 100  # Convert to percentage
    severity_scale_bar['value'] = severity_scale  # Update the progress bar

def plot_confidence_graph():
    image = Image.open(upload_image())
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()

    labels = [
        "No DR signs", "Mild NPDR", "Moderate NPDR", 
        "Severe NPDR", "Very Severe NPDR", "PDR", "Advanced PDR"
    ]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, probs, color="skyblue")
    plt.xlabel('Confidence (%)')
    plt.title('Confidence Distribution for Diabetic Retinopathy Severity')
    plt.xlim(0, 1)
    plt.show()

# Initialize the root window
root = tk.Tk()
root.title("Diabetic Retinopathy Detection")
root.geometry("450x500")
root.configure(bg="#e8f1fa")

# Header label with project details
title_label = ttk.Label(root, text="Diabetic Retinopathy Detection System", font=('Helvetica', 16, 'bold'), background="#e8f1fa")
title_label.pack(pady=15)

# Display course and team details
course_label = ttk.Label(root, text="Capstone Project - FALL SEM 2024-2025", font=('Helvetica', 10), background="#e8f1fa")
course_label.pack(pady=5)

team_label = ttk.Label(root, text="TEAM MEMBERS: \nN. KAMALESH 21BCE7751\nB. SIRI 21BCE7232\nV. ABHIRAM 21BCE7740", font=('Helvetica', 10), background="#e8f1fa")
team_label.pack(pady=5)

# Create labels and entry fields
ttk.Label(root, text="Username:").pack(pady=5)
username_entry = ttk.Entry(root, width=30)
username_entry.pack(pady=5)

ttk.Label(root, text="Password:").pack(pady=5)
password_entry = ttk.Entry(root, show="*", width=30)
password_entry.pack(pady=5)

# Create buttons
button_frame = ttk.Frame(root, style='TFrame')
button_frame.pack(pady=20)

ttk.Button(button_frame, text="Sign Up", command=signup).grid(row=0, column=0, padx=10)
ttk.Button(button_frame, text="Login", command=login).grid(row=0, column=1, padx=10)

# Main window to show severity scale progress bar and image upload button
def main_window():
    global result_label, severity_scale_bar
    root.withdraw()
    main = tk.Toplevel()
    main.title("Upload Image for Prediction")
    main.geometry("550x400")
    main.configure(bg="#e8f1fa")

    ttk.Label(main, text="Upload an image to get a prediction:", font=('Helvetica', 13)).pack(pady=15)
    ttk.Button(main, text="Upload Image", command=upload_image).pack(pady=10)

    result_label = ttk.Label(main, text="Prediction: ", font=('Helvetica', 12))
    result_label.pack(pady=15)

    # Create and display severity scale progress bar
    severity_scale_bar = ttk.Progressbar(main, orient="horizontal", length=300, mode="determinate", maximum=100)
    severity_scale_bar.pack(pady=20)

root.mainloop()
