import torch
from PIL import Image
from torchvision import transforms
from cnn import FERModel

# Load the model
model = FERModel(num_classes=7)  # replace with your number of classes
# replace with your model path
model.load_state_dict(torch.load('model_42.pth'))
model.eval()

# Load the image
image = Image.open('image2.jpg')  # replace with your image path

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Preprocess the image
image = transform(image)
image = image.unsqueeze(0)  # add a batch dimension

# Make the prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print('Predicted class:', predicted.item())
