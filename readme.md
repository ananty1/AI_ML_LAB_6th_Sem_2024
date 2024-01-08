# MNIST Classifier Web App

This project implements a simple web application for classifying handwritten digits using a Convolutional Neural Network (CNN). The app is built with Streamlit and PyTorch.

## Getting Started

### Prerequisites

- Python 3.x
- Conda (optional but recommended)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ananty1/mnist_classifier_app.git
    cd mnist_classifier_app
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    conda create --name mnist_classifier_env python=3.x
    conda activate mnist_classifier_env
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Train the Model

Make sure to train the model if it's not already trained:

    ```bash
    python src/model.py
    ```



### Run the App
To start the web application, run the following command:

    ```bash
    streamlit run src/app.py
    ```
Visit localhost:8501 in your web browser to interact with the app.


Fine-Tuning the Model
If you want to fine-tune the pre-trained model, run:

    ```bash
    python src/Fine_Tune_Model.py
    ```