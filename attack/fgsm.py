# fgsm.py
import torch
import torch.nn.functional as F

class FGSMAttack:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def generate(self, model, image, label):
        """
        Generate adversarial example using FGSM.

        Args:
        - model: The pre-trained model.
        - image: The input image (Tensor).
        - label: The correct label (Tensor).

        Returns:
        - adversarial_image: The generated adversarial image.
        """
        # Make the image require gradients
        image.requires_grad = True

        # Forward pass through the model
        output = model(image)

        # Compute the loss
        loss = F.cross_entropy(output, label)
        
        # Backpropagate to compute gradients
        model.zero_grad()
        loss.backward()

        # Collect the gradients of the input image
        gradient = image.grad.data

        # Create the adversarial image by adding the perturbation
        adversarial_image = image + self.epsilon * gradient.sign()

        # Ensure the adversarial image is still within the valid range [0, 1]
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        return adversarial_image
