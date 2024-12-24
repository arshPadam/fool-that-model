import torch
import torch.nn.functional as F

def calculate_loss(model, image, label):
    output = model(image)  # Get model output
    loss = F.cross_entropy(output, label)  # Calculate cross-entropy loss
    return loss.item()

def calculate_accuracy(model, image, label):
    output = model(image)  # Get model output
    _, predicted = torch.max(output, 1)  # Get predicted class
    correct = (predicted == label).sum().item()  # Count correct predictions
    accuracy = correct / label.size(0)  # Calculate accuracy
    return accuracy
