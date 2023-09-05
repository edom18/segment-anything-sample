from flask import Flask, request
import os
import io
import cv2
import datetime

from recognizer.segument import SAM

def generate_date_filename():
    # 現在の日時を取得
    now = datetime.datetime.now()
    # YYYYMMDD_HHMMSS 形式の文字列を生成
    return now.strftime('%Y%m%d_%H%M%S')


app = Flask(__name__)

sam = SAM()

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
    
    if not file:
        return "No file found"

    filename = os.path.join(UPLOAD_FOLDER, f"received_image_{generate_date_filename()}.jpg")
    file.save(filename)

    result = sam.crop(filename)

    is_success, buffer = cv2.imencode('.jpg', result)
    if not is_success:
        return "Failed to encode image"


    cv2.imwrite(f'outputs/result_{generate_date_filename()}.jpg', result)

    return "Success"
    # io_buf = io.BytesIO(buffer)

    # return io_buf.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
