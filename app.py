from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "5555555555"  # Change this to a random secret key

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your trained model
scratch_model = load_model("models/scratch.keras")
resnet_model = load_model("models/resnet-basic.keras")
resnet_tuned_model = load_model("models/resnet-tuned.keras")
vgg_model = load_model("models/vgg16_final_model.keras")
effnet_model = load_model("models/effnet.keras")

# Assuming your model has a property for class names
class_names = ("glioma", "meningioma", "notumor", "pituitary")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Create a form class
class UploadForm(FlaskForm):
    file = FileField("Image File", validators=[DataRequired()])
    submit = SubmitField("Upload")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_path):
    img = Image.open(image_path)
    # Resize to match the model input shape (150x150)
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
    return img_array


def prepare_effnet(image_path):
    img = Image.open(image_path)
    # Resize to match the model input shape (150x150)
    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
    return img_array


def predict_tumor(model, img, name):
    prediction = model.predict(img)
    predicted_class = np.argmax(
        prediction, axis=1
    )  # Get the index of the predicted class
    confidence = round(np.max(prediction) * 100, 2)
    if name == "Model from scratch" or name == "VGG16":
        prediction = softmax(prediction)
        confidence = round(np.max(prediction) * 100, 2)
    predicted_label = class_names[predicted_class[0]]
    flash(
        f"""{name}: {predicted_label} \t|\tConfidence: {
            confidence}%""",
        "success",
    )


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            img_array = prepare_image(file_path)
            img_array2 = prepare_effnet(file_path)

            predict_tumor(scratch_model, img_array, "Model from scratch")

            predict_tumor(resnet_model, img_array, "ResNet")

            predict_tumor(resnet_tuned_model, img_array, "ResNet Tuned")

            predict_tumor(effnet_model, img_array2, "Effnet")

            predict_tumor(vgg_model, img_array, "VGG16")

            return redirect(url_for("index"))
        else:
            flash("File not allowed or invalid format", "danger")
            return redirect(request.url)

    # Pass the form to the template
    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
