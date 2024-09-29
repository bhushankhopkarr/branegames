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
base_model = load_model("models/resnet-basic.keras")
tuned_model = load_model("models/resnet-tuned.keras")

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
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()  # Create an instance of the form
    if form.validate_on_submit():  # Check if form is submitted and valid
        file = form.file.data  # Get the uploaded file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Prepare the image and make a prediction
            img_array = prepare_image(file_path)

            scratch_prediction = scratch_model.predict(img_array)
            print("scratch prediction:", scratch_prediction)  # Debugging line
            predicted_class = np.argmax(
                scratch_prediction, axis=1
            )  # Get the index of the predicted class
            scratch_predicted_label = class_names[predicted_class[0]]

            base_prediction = base_model.predict(img_array)
            print("base prediction:", base_prediction)  # Debugging line
            predicted_class = np.argmax(
                base_prediction, axis=1
            )  # Get the index of the predicted class
            base_predicted_label = class_names[predicted_class[0]]

            tuned_prediction = tuned_model.predict(img_array)
            print("tuned prediction:", tuned_prediction)  # Debugging line
            predicted_class = np.argmax(
                tuned_prediction, axis=1
            )  # Get the index of the predicted class
            tuned_predicted_label = class_names[predicted_class[0]]

            flash(f"Scratch Prediction: {scratch_predicted_label}", "success")
            flash(f"Base Prediction: {base_predicted_label}", "success")
            flash(f"Tuned Prediction: {tuned_predicted_label}", "success")
            return redirect(url_for("index"))
        else:
            flash("File not allowed or invalid format", "danger")
            return redirect(request.url)

    # Pass the form to the template
    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
