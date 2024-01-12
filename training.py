import torch
from torch import optim, nn
from cnnNew import FERModel
from preprocessing import train_loader, test_loader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

# Create an instance of the model
num_classes = 7  # replace with the number of classes in your dataset
model = FERModel(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train for
num_epochs = 50

# Loop over the dataset multiple times
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=int(len(train_loader)))
    for i, data in progress_bar:
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        progress_bar.set_description(
            f"Epoch {epoch + 1} loss: {running_loss/(i+1)}")

# Save the trained model
torch.save(model.state_dict(), 'bestmodel.pth')

# Evaluate the model on the test set and print a classification report
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

print(classification_report(all_labels, all_predictions))

print('Finished Training')
