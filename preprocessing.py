import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torch.utils.data import ConcatDataset

IMG_HEIGHT = 48
IMG_WIDTH = 48

# Path to the training data
TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'data', 'train')

# Path to the test data
TEST_DATA_PATH = os.path.join(os.getcwd(), 'data', 'test')

# Define your transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

# Load the datasets
train_dataset = ImageFolder(TRAIN_DATA_PATH, transform=transform)
test_dataset = ImageFolder(TEST_DATA_PATH, transform=transform)

# Create the dataloader for validation set only, train data still needs to be augmented
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Oversampling the disgust samples since we don't have many samples

# Define additional transformations for data augmentation
augment_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Create a new dataset with only the "disgust" images
disgust_dataset = [img for img in train_dataset if img[1]
                   == train_dataset.class_to_idx['disgust']]

# Convert Tensor to PIL Image
to_pil = ToPILImage()

# Apply data augmentation to the "disgust" images
augmented_disgust_dataset = [(augment_transform(to_pil(img[0])), img[1])
                             for _ in range(8) for img in disgust_dataset]

# Combine the original dataset with the augmented "disgust" images
train_dataset = ConcatDataset([train_dataset, augmented_disgust_dataset])

# Update the train DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
