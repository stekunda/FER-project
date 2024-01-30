import torch
from PIL import Image
from torchvision import transforms
from cnn import FERModel

# Load the model architecture
model = FERModel(num_classes=7)

# Load the trained model weights
model.load_state_dict(torch.load('model_42.pth'))
model.eval()

# Load the image to run prediction on
image = Image.open('image2.jpg')

# Define the transformations for preprocessing the image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Preprocess the image
image = transform(image)
image = image.unsqueeze(0)

# Make the prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print('Predicted class:', predicted.item())
