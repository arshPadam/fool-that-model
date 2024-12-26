# fgsm.py
import torch
import torch.nn.functional as F

class FGSMAttack:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def generate(self, model, image, label):
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

        return adversarial_image
