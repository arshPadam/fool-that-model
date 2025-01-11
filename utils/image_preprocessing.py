
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
    # Ensure tensor is on CPU and detached from the computation graph if it's a tensor that requires gradients
    image_tensor = image_tensor.detach().cpu()

    # Check if the tensor has a batch dimension, and remove it if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension, Shape: [C, H, W]
    
    # Mean and std values for denormalization (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406])  # Mean values for R, G, B channels
    std = torch.tensor([0.229, 0.224, 0.225])  # Inverse std values for R, G, B channels

    # Denormalize each channel by multiplying with std and adding mean
    # Broadcasting std and mean for image_tensor dimensions
    denormalized_tensor = image_tensor * std[:, None, None] + mean[:, None, None]

    # Clamp values to the [0, 1] range to avoid invalid pixel values
    denormalized_tensor = denormalized_tensor.clamp(0, 1)

    # Convert the tensor to a numpy array and change the shape to HWC for PIL
    image = denormalized_tensor.permute(1, 2, 0).detach().numpy()  # Shape: [H, W, C]
    
    # Convert to [0, 255] range and uint8
    image = (image * 255).astype(np.uint8)
    
    # Convert the numpy array to a PIL image
    image_pil = Image.fromarray(image)
    
    # Save the image to the specified path
    image_pil.save(save_path)
    print(f"Image saved to {save_path}")
