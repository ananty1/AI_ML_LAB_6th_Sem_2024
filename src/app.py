import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import MNISTClassifier
import os
import pandas as pd
import numpy as np 

# Define transformations for image preprocessing for MNIST
mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the trained MNIST model
mnist_model = MNISTClassifier()
mnist_model.load_state_dict(torch.load("model/mnist_classifier.pth"))
mnist_model.eval()

# Function to make predictions for MNIST
def mnist_predict(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = mnist_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = mnist_model(image)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()


# Function to store incorrect predictions
def store_incorrect_prediction(image, predicted_class, selected_label):
    data = {
        'Image_Path': [f'incorrect_predictions/correct_{selected_label}_predicted_{predicted_class}.png'],
        'Predicted_Class': [predicted_class],
        'Selected_Label': [selected_label]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV file (you can replace this with database storage)
    csv_path = "incorrect_predictions/incorrect_predictions.csv"
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    
     # Save the input image to the folder
    image.save(f"incorrect_predictions/correct_{selected_label}_predicted_{predicted_class}.png")


#Streamlit app
st.title("MNIST Digit Classifier")
# Create a folder to store incorrect predictions
if not os.path.exists("incorrect_predictions"):
    os.makedirs("incorrect_predictions")
uploaded_file = st.file_uploader("Choose a digit image...",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = Image.open(uploaded_file)
    # Perform prediction
    class_index = mnist_predict(uploaded_file)

    # Display the result and handle feedback
    st.write(f"Predicted Digit: {class_index}")

    # Ask the user if the prediction is correct
    user_feedback = st.radio("Is the prediction correct?", ("Yes", "No"))

    if user_feedback == "Yes":
        st.write("Check for other Images")
    else:
        # Ask the user for the correct output in the range of 0-9
        correct_output = st.number_input("Enter the correct output (between 0 and 9)", min_value=0, max_value=9, step=1)

        # Process the correct output
        st.write(f"Thank you for providing feedback. The correct output is: {correct_output}")

        
        # # Submit button for user feedback
        if st.button("Submit Feedback"):
            # Store incorrect prediction
            store_incorrect_prediction(image, class_index, correct_output)
            st.success("Incorrect prediction stored successfully!")

