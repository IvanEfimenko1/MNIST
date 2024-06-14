# MNIST Neural Network Project

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The implementation is done using PyTorch, and the project includes data preprocessing, model training, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Docker (optional, for containerized execution)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/IvanEfimenko1/MNIST_Chat_Bot.git
    cd MNIST_Chat_Bot
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If you prefer to use Docker, build the Docker container:

    ```bash
    docker-compose build
    ```

## Running the Notebook

1. Activate the virtual environment if not already activated:

    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the `Caggleton_7.ipynb` notebook and run the cells sequentially to train and evaluate the model.

## Project Structure

- `.dockerignore`: Specifies files and directories to be ignored by Docker.
- `Caggleton_7.ipynb`: Jupyter notebook containing the implementation and experiments with the MNIST dataset.
- `docker-compose.yml`: Docker Compose file for managing multi-container Docker applications.
- `Dockerfile`: Instructions for building the Docker image.
- `README.md`: This file, provides information about the project.
- `requirements.txt`: List of project dependencies.
- `MNIST/`: Directory containing MNIST dataset files.
  - `train-labels.idx1-ubyte`: Training labels.
  - `t10k-labels.idx1-ubyte`: Test labels.

## Data Preprocessing

The data preprocessing steps include:
- Normalization of the images to have pixel values between -1 and 1.
- Conversion of images to PyTorch tensors.

Example code:

```python
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
Model Architecture
The Convolutional Neural Network (CNN) architecture used in this project includes:

Convolutional layers
Max pooling layers
Fully connected layers
Dropout layers for regularization
Example code:

python
Копировать код
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
Training
The training process involves:

Defining the loss function and optimizer
Running multiple epochs with forward and backward propagation
Calculating the training and validation accuracy and loss
Example code:

python
Копировать код
import torch.optim as optim

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
Evaluation
The evaluation process includes:

Running the trained model on the test dataset
Calculating the accuracy and generating the classification report
Example code:

python
Копировать код
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))

