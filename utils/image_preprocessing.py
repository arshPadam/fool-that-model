
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    try:
        # Open the image and ensure it's in RGB mode
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the size expected by the model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet
    ])
    
    # Apply the transformations and add a batch dimension
    image_tensor = preprocess(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    return image_tensor


import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def denormalize_and_save_image(image_tensor, save_path):
    # Remove the batch dimension
    image_tensor = image_tensor.squeeze(0)  # Shape: [3, H, W]
    
    # Convert to HWC format (Height, Width, Channels)
    image_tensor = image_tensor.permute(1, 2, 0)
    
    # Denormalize the image using the ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    image_tensor = image_tensor * std + mean
    
    # Clip the values to ensure they are in the valid range [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to numpy array
    image = image_tensor.detach().numpy()
    
    # Convert the numpy array to a PIL image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Convert to [0, 255] range and uint8
    
    # Save the image
    image_pil.save(save_path)

