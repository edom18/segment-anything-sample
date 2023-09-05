from flask import Flask, request
import os

import recognizer.segument

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = os.path.join(UPLOAD_FOLDER, "received_image.jpg")
        file.save(filename)
        return "Image saved!"

if __name__ == '__main__':
    app.run(debug=True)
