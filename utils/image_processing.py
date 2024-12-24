from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    image = Image.open(image_path)

    # Define the transformations: resizing, converting to tensor, and normalization
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image
