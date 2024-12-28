import torch
import torch.nn.functional as F
import numpy as np

def deepfool(model, X, num_classes=10, max_iter=50, epsilon=1e-4):
    """
    DeepFool attack implementation to generate adversarial examples.
    
    Parameters:
    - model: The trained model (PyTorch).
    - X: The input image (as a tensor).
    - num_classes: The number of classes in the classification problem.
    - max_iter: The maximum number of iterations to apply the attack.
    - epsilon: A small number to avoid division by zero in case of close gradients.
    
    Returns:
    - perturbed_image: The adversarial image after applying DeepFool.
    - perturbation: The perturbation added to the original image.
    """
    # Set the model to evaluation mode
    model.eval()

    # Ensure the input image is a tensor and requires gradient
    image = X.clone().detach().requires_grad_(True)
    
    # Get the true class label
    true_label = torch.argmax(model(X), dim=1).item()
    
    perturbation = torch.zeros_like(X)

    for _ in range(max_iter):
        # Forward pass
        output = model(image)
        
        # Compute the loss with respect to the true label
        true_class_score = output[0, true_label]
        
        # Compute gradients with respect to the loss (cross-entropy with one-hot encoding)
        model.zero_grad()
        true_class_score.backward(retain_graph=True)
        
        # Get the gradients of the image
        gradients = image.grad.data

        # Compute the perturbation
        perturbation_direction = torch.sign(gradients)
        perturbation += perturbation_direction

        # Update the image with the perturbation
        perturbed_image = X + perturbation
        
        # Ensure the perturbed image is within valid range [0, 1]
        #perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Check if the adversarial image causes a misclassification
        if torch.argmax(model(perturbed_image), dim=1).item() != true_label:
            break
    
    return perturbed_image, perturbation

# Example usage:
# Assume `model` is your trained PyTorch model, and `X` is the input image tensor.
# X should have shape [batch_size, channels, height, width]
# perturbed_image, perturbation = deepfool(model, X)
