# Fool That Model - Adversarial Attack Application

Fool That Model is an adversarial attack application designed to demonstrate how adversarial examples can be created to deceive machine learning models. It applies the **Fast Gradient Sign Method (FGSM)** to perturb an input image in order to cause misclassification by a pre-trained model. The model used for this project is **ResNet18**, trained on the **ImageNet** dataset. The application takes in an image, perturbs it, and returns the resulting image that causes misclassification.

## Sample Results

Here is an example of an input image and the resulting adversarial image generated using the FGSM attack:

**Input Image:** Classified as Lion 

![Input Image](https://github.com/arshPadam/fool-that-model/blob/master/images/input_images/input.jpg)

**Adversarial Image:** Classified as papillon

![Adversarial Image](https://github.com/arshPadam/fool-that-model/blob/master/images/adversarial_images/output.jpg)


## Features
- Takes an input image.
- Applies the **FGSM** attack to generate adversarial examples.
- Uses **ResNet18** model pre-trained on the **ImageNet** dataset.
- Outputs the adversarial image designed to mislead the model.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- Pillow
- numpy

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/fool-that-model.git
cd fool-that-model
pip install -r requirements.txt
