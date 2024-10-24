import os
from wtforms.validators import DataRequired
from wtforms import FileField, SubmitField
from flask_wtf import FlaskForm
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Randomize secret key for security

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Class names for the predictions
class_names = ("glioma", "meningioma", "notumor", "pituitary")

# Create a form class for file upload
class UploadForm(FlaskForm):
    file = FileField("Image File", validators=[DataRequired()])
    submit = SubmitField("Upload")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your trained models once (Lazy loading to improve startup time)
def load_models():
    models = {
        "scratch": keras.models.load_model("models/scratch.keras"),
        "resnet": keras.models.load_model("models/resnet-basic.keras"),
        "resnet_tuned": keras.models.load_model("models/resnet-tuned.keras"),
        "vgg": keras.models.load_model("models/vgg16_final.keras"),
        "effnet": keras.models.load_model("models/effnet-basic.keras"),
        "effnet_tuned": keras.models.load_model("models/effnet-tuned.keras"),
    }
    return models

# Prepare image for prediction
def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
    return img_array

# Single model prediction
def predict_tumor(model, img, model_name):
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
    confidence = round(np.max(prediction) * 100, 2)
    
    # Apply softmax for models that require it
    if model_name in ["Model from scratch", "VGG16"]:
        prediction = softmax(prediction)
        confidence = round(np.max(prediction) * 100, 2)

    predicted_label = class_names[predicted_class]
    flash(f"{model_name}: {predicted_label} | Confidence: {confidence}%", "success")
    return predicted_class

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Calculate ensemble prediction
def ensemble_prediction(predictions):
    ensemble_pred = np.bincount(predictions).argmax()  # Majority voting
    ensemble_confidence = round(predictions.count(ensemble_pred) * 100 / len(predictions), 2)
    predicted_label = class_names[ensemble_pred]
    flash(f"Ensemble: {predicted_label} | Confidence: {ensemble_confidence}%", "info")

@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    models = load_models()  # Load models

    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Prepare the image
            img_array = prepare_image(file_path)

            # Make predictions from each model
            predictions = [
                predict_tumor(models["scratch"], img_array, "Model from scratch"),
                predict_tumor(models["resnet"], img_array, "ResNet50"),
                predict_tumor(models["resnet_tuned"], img_array, "ResNet50 Tuned"),
                predict_tumor(models["vgg"], img_array, "VGG16"),
                predict_tumor(models["effnet"], img_array, "EfficientNetB0"),
                predict_tumor(models["effnet_tuned"], img_array, "EfficientNetB0 Tuned"),
            ]

            # Calculate ensemble prediction
            ensemble_prediction(predictions)
            return redirect(url_for("index"))
        else:
            flash("Invalid file format. Please upload a PNG, JPG, or JPEG file.", "danger")
            return redirect(request.url)

    return render_template("index.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)