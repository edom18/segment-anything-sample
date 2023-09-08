from flask import Flask, request
import os
import io
import cv2
import datetime
import json

from recognizer.segument import SAM
from recognizer.caption import ImageChecker
from color_checker.checker import ColorData, check

def generate_date_filename():
    # 現在の日時を取得
    now = datetime.datetime.now()
    # YYYYMMDD_HHMMSS 形式の文字列を生成
    return now.strftime('%Y%m%d_%H%M%S')

app = Flask(__name__)

# sam = SAM()
checker = ImageChecker()

config = json.loads(open('configs/label_config.json').read())

missions = [mission for mission in config['missions']]

color_data = ColorData(
    labels=[color['label'] for color in config['colors']],
    rgbs=[color['elements'] for color in config['colors']]
)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():

    print('received request')

    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if not file:
        return "No file found"

    filename = os.path.join(UPLOAD_FOLDER, f"received_image_{generate_date_filename()}.jpg")
    file.save(filename)

    # result = sam.crop(filename)

    # is_success, buffer = cv2.imencode('.jpg', result)
    # if not is_success:
    #     return "Failed to encode image"

    # output_path = f'outputs/result_{generate_date_filename()}.jpg'
    # cv2.imwrite(output_path, result)

    text, score = checker.check(filename, missions)

    result_index = check(filename, color_data, debug=False)
    result_label = color_data.getlabelAt(result_index)

    response = {
        "object": {
            text: float(score),
        },
        "color": result_label,
    }

    return json.dumps(response)

    # io_buf = io.BytesIO(buffer)
    # return io_buf.getvalue()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
