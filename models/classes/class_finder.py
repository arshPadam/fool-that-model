import json

def class_to_label(label):

    with open('models\\classes\\imagenet_classes.json', 'r') as f:
        imagenet_classes = json.load(f)

    class_id = f"{label}"
    print(f"Class {label} corresponds to: {imagenet_classes[class_id]}")