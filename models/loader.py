# loader.py
import torch
from torchvision import models

def load_model(model_name="resnet18"):
    """
    Loads a pre-trained model from torchvision based on the model name.

    Args:
    - model_name (str): Name of the model to load (e.g., 'resnet18', 'vgg16').

    Returns:
    - model (nn.Module): The pre-trained model.
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.eval()  # Set the model to evaluation mode
    return model
