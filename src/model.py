import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms
import pandas as pd
from torchvision.io import read_image
import numpy as np 



class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(self, root_path: str):
        # Define transformations for data augmentation and normalization
        self.transform = transforms.Compose(
            [
                # transforms.Resize((28, 28)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        
        self.root = root_path
        self.threshold = 0.5
        # Initialize the model, loss function, and optimizer
        self.model = MNISTClassifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _load_data(self):
        train_data = pd.read_csv("input/train/train.csv")
        Y_data = np.array(train_data["label"])
        X_data = np.array(train_data.loc[:,"pixel0":]).reshape(-1,1,28,28)/255

        tensor_x = torch.from_numpy(X_data).float()
        tensor_y = torch.from_numpy(Y_data)

        my_dataset = TensorDataset(tensor_x,tensor_y)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(my_dataset, (int(len(X_data)*0.99),int(len(X_data)*0.01)))

        print("The train_images looks like",self.train_dataset)
        print("LOading works....")

    def _extract_data(self):
        print("Extraction starts")
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64)
        print("Extracting works...")

    def _eval(self):
        self.model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in self.val_loader:
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(
                    val_outputs, val_labels #.type(torch.FloatTensor)
                )

                total_val_loss += val_loss.item()
                val_predicted = torch.argmax(val_outputs, dim=1)
                correct_val_predictions += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        return val_loss, val_accuracy

    def _train(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def train(self, num_epochs: int = 5):
        self._load_data()
        self._extract_data()
        # print("I am goinf to run train....")

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train()
            print(
                f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
            val_loss, val_accuracy = self._eval()
            print(
                f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )

    def save(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    trainer = Trainer(root_path="input")
    trainer.train()
    trainer.save(save_path="model/mnist_classifier.pth")