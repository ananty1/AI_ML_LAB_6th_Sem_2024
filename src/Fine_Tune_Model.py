import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd 
from PIL import Image
from app import MNISTClassifier

#We have a dataset class to load incorrect predictions from the CSV file
class IncorrectDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            # Your image transformations if needed
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'Image_Path']
        label = self.df.loc[idx, 'Selected_Label']
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        return image, label

# Load the pre-trained model
mnist_model = MNISTClassifier()
mnist_model.load_state_dict(torch.load("model/mnist_classifier.pth"))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mnist_model.parameters(), lr=0.001, momentum=0.9)

# Load incorrect predictions data
incorrect_dataset = IncorrectDataset("incorrect_predictions/incorrect_predictions.csv")
incorrect_dataloader = DataLoader(incorrect_dataset, batch_size=32, shuffle=True)

# Fine-tune the model
num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    for images, labels in incorrect_dataloader:
        optimizer.zero_grad()

        # Print the shape of the input tensor
        print(f"Input shape before: {images.shape}")

        outputs = mnist_model(images)

        # Print the shape of the output tensor
        print(f"Output shape: {outputs.shape}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Save the fine-tuned model
torch.save(mnist_model.state_dict(), "model/fine_tuned_mnist_classifier.pth")
