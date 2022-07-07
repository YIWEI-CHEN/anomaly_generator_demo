from datetime import datetime

import cv2
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

from anomaly_generator import AnomalyGenerator

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 16 * 1000 * 1000
app.secret_key = 'secret_key'

app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            name = filename.split('.')[0]
            aug_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"aug_{name}_{timestamp}")
            os.makedirs(aug_dir, exist_ok=True)
            generator = AnomalyGenerator(img_path, resize_shape=(256, 256))
            aug_img_paths = []
            for style, aug_func in (
                ("Perlin Noise", generator.perlin_noise),
                ("Cutout", generator.cutout),
                ("Scar", generator.scar),
                ("Cutpaste", generator.cutpaste),
                ("Cutpaste Scar", generator.cutpaste_scar),
            ):
                _, aug_img = aug_func()
                aug_img_path = os.path.join(aug_dir, f"{name}_{style}.png")
                cv2.imwrite(aug_img_path, aug_img)
                aug_img_paths.append((style, '/'.join(aug_img_path.split('/')[1:])))
            # return redirect(url_for('download_file', name=filename))
            flash('Image successfully uploaded and displayed below')
            return render_template('upload.html', filename=filename, aug_img_paths=aug_img_paths)
    return render_template('upload.html')


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=os.path.join('uploads', filename)), code=301)


if __name__ == '__main__':
    app.run()

