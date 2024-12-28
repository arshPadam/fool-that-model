
import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf


from models.loader import load_model
from models.evaluator import evaluate_model
from models.evaluator import get_label
from utils.image_preprocessing import load_and_preprocess_image
from utils.image_preprocessing import denormalize_and_save_image
from attacks.fgsm import FGSMAttack
from attacks.deepfool import deepfool
from attacks.ead import ead_attack
from models.classes.class_finder import class_to_label

def main():
    # Step 1: Load the model
    model_name = "resnet18"  # Specify the model name
    model = load_model(model_name)
    
    # Step 2: Load and preprocess the image
    image_path = "cat.jpg"  # Path to the image
    image_tensor = load_and_preprocess_image(image_path)
    
    if image_tensor is None:  # If the image could not be loaded, exit early
        return
    
    # Step 3: Get a new label (predicted class) using the get_new_label function
    predicted_label = get_label(model, image_tensor)
    adversarial_image_tensor = [0]
    str = "ead"
    
    if(str == "fgsm"):
        fgsm_attack = FGSMAttack(epsilon=0.1)
        adversarial_image_tensor = fgsm_attack.generate(model, image_tensor, torch.tensor([predicted_label]))       
    elif(str == "deepfool"):
        adversarial_image_tensor,_ = deepfool(model, image_tensor)
    elif(str == "ead"):
        adversarial_image_tensor, _ = ead_attack(model, image_tensor, torch.tensor([predicted_label]), epsilon=0.1, max_iter=100, lambda_1=0.1, lambda_2=0.1)
    
    print(adversarial_image_tensor)
    print(image_tensor)
        
    adversarial_label = get_label(model, adversarial_image_tensor)
    denormalize_and_save_image(image_tensor, "images\\input_images\\input.jpg")
    denormalize_and_save_image(adversarial_image_tensor, "images\\adversarial_images\\output.jpg")
   
    # Step 4: Display the result
    print(f"Predicted class index: {predicted_label}")
    print(f"Aversarial class index: {adversarial_label}")
    class_to_label(predicted_label)
    class_to_label(adversarial_label)

if __name__ == "__main__":
    main()
