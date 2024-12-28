import torch
import torch.nn.functional as F
import numpy as np

def ead_attack(model, X, y, epsilon=0.1, max_iter=100, lambda_1=0.1, lambda_2=0.1, max_perturbation=0.5):
    """
    Elastic Net Adversarial Attack (EAD) with controlled perturbation magnitude.
    
    Parameters:
    - model: The trained model (PyTorch).
    - X: The input image tensor (with shape [batch_size, channels, height, width]).
    - y: The true label for the image.
    - epsilon: The perturbation step size (controls the size of perturbation at each step).
    - max_iter: Maximum number of iterations for the optimization process.
    - lambda_1: Weight for L1 regularization (sparsity).
    - lambda_2: Weight for L2 regularization (smoothness).
    - max_perturbation: Maximum allowed perturbation magnitude (prevents large changes).
    
    Returns:
    - perturbed_image: The adversarial image after applying the attack.
    - perturbation: The perturbation added to the original image.
    """
    
    # Set the model to evaluation mode
    model.eval()

    # Ensure the input image is a tensor and requires gradient
    image = X.clone().detach().requires_grad_(True)
    
    # Get the true label index for the target class
    true_label = y

    # Optimizer for the perturbation
    optimizer = torch.optim.Adam([image], lr=epsilon)

    for iteration in range(max_iter):
        # Forward pass through the model
        output = model(image)
        
        # Compute the loss (cross-entropy)
        loss = F.cross_entropy(output, true_label)

        # Add the regularization terms (L1 and L2)
        l1_norm = torch.sum(torch.abs(image))
        l2_norm = torch.sum(torch.pow(image, 2))

        total_loss = loss + lambda_1 * l1_norm + lambda_2 * l2_norm

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Update the image with the gradient
        optimizer.step()

        # Clip the image to stay within valid pixel range [0, 1]
        #image.data = torch.clamp(image.data, 0, 1)

        # Compute the perturbation
        perturbation = image - X
        
        # Apply maximum perturbation constraint
        perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
        
        # Add the perturbation back to the image to form the adversarial image
        image.data = X + perturbation

        # Check if the adversarial image misclassifies
        if torch.argmax(model(image), dim=1).item() != true_label.item():
            break

    return image, perturbation

# Example usage:
# Assuming 'model' is your trained PyTorch model and 'X' is the input image tensor
# X should have shape [batch_size, channels, height, width], and y should be the label for the image

