# evaluator.py
import torch
import torch.nn.functional as F

def get_label(model, image):
    with torch.no_grad():  # Disable gradient computation during inference
        output = model(image)  # Get the model's output
        _, predicted = torch.max(output, 1)  # Get the predicted class index
    return predicted.item()  # Return the predicted class label as a scalar (integer)


def evaluate_model(model, image, label):
    with torch.no_grad():  # Disable gradient computation
        output = model(image)  # Get the model's output
        _, predicted = torch.max(output, 1)  # Get the predicted class
        correct = (predicted == label).sum().item()  # Count the correct predictions
        accuracy = correct / label.size(0)  # Compute accuracy
    return accuracy
