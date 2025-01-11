
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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


def denormalize_and_save_image(image_tensor, save_path):
    # Check if the tensor has a batch dimension, and remove it if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension, Shape: [C, H, W]
    
    # Ensure the tensor is in the correct format [C, H, W] for denormalization
    if image_tensor.dim() == 3 and image_tensor.size(0) == 3:
        # Inverse normalization: undo the normalization using mean and std
        inv_normalize = transforms.Normalize(
            mean=[-0.485, -0.456, -0.406], 
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        image_tensor = inv_normalize(image_tensor).clamp(0, 1)  # Denormalize and ensure pixel values are within [0, 1]
    else:
        raise ValueError("Input tensor is not in the expected format [C, H, W].")
    
    # Convert the tensor to a numpy array, then scale back to the [0, 255] range
    image = image_tensor.detach().numpy().transpose(1, 2, 0)  # Convert to [H, W, C] format
    
    # Convert the numpy array to a PIL image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Convert to [0, 255] range and uint8
    
    # Save the image to the specified path
    image_pil.save(save_path)
    print(f"Image saved to {save_path}")