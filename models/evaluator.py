# evaluator.py
import torch
import torch.nn.functional as F

def evaluate_model(model, image, label):
    """
    Evaluates a model's prediction accuracy on a given image.

    Args:
    - model (nn.Module): The pre-trained model to evaluate.
    - image (Tensor): The input image.
    - label (Tensor): The true label for the image.

    Returns:
    - accuracy (float): The accuracy of the model on the given image.
    """
    with torch.no_grad():  # Disable gradient computation
        output = model(image)  # Get the model's output
        _, predicted = torch.max(output, 1)  # Get the predicted class
        correct = (predicted == label).sum().item()  # Count the correct predictions
        accuracy = correct / label.size(0)  # Compute accuracy
    return accuracy
