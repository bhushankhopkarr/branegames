# Brain Tumor Detection

This repository contains a Flask web application for detecting brain tumors from MRI scans using Convolutional Neural Networks (CNN). The application allows users to upload MRI images and get predictions from multiple pre-trained models.

## Project Structure

- **app.py**: Main Flask application file.
- **confusion_matrices/**: Directory for storing confusion matrices.
- **requirements.txt**: List of dependencies required for the project.
- **static/**: Directory for static assets like images and CSS.
- **templates/**: Directory for HTML templates.
- **Testing.ipynb**: Jupyter notebook for testing and evaluating models.
- **uploads/**: Directory for storing uploaded MRI images.

## Usage
1. Upload an MRI image using the form on the homepage.
2. The application will process the image and provide predictions from multiple models.
3. The results will be displayed on the webpage, including the predicted class and confidence level.

## Models
The application uses the following pre-trained models:

1. Scratch model: ``models/scratch.keras``
2. ResNet: ``models/resnet-basic.keras``
3. Tuned ResNet: ``models/resnet-tuned.keras``
4. VGG16: ``models/vgg16_final.keras``
5. EfficientNet: ``models/effnet-basic.keras``
6. Tuned EfficientNet: ``models/effnet-tuned.keras``

## Acknowledgements
- TensorFlow
- Flask
- Keras
- PIL
