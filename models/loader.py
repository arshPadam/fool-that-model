# loader.py
import torch
from torchvision import models

def load_model(model_name="resnet18"):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.eval()  # Set the model to evaluation mode
    return model
