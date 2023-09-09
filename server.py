from flask import Flask, request
import os
import io
import cv2
import datetime
import json

# from recognizer.segument import SAM
from recognizer.caption import ImageChecker
from color_checker.checker import ColorData, check
from typing import List
from typing import Dict

def generate_date_filename():
    # 現在の日時を取得
    now = datetime.datetime.now()
    # YYYYMMDD_HHMMSS 形式の文字列を生成
    return now.strftime('%Y%m%d_%H%M%S')

app = Flask(__name__)

# sam = SAM()
checker = ImageChecker()

# config = json.loads(open('configs/label_config.json').read())

def validation(data: Dict) -> bool:
    
        if 'missions' not in data:
            return False
    
        if 'colors' not in data:
            return False

        for color in data['colors']:
            if 'label' not in color:
                return False
            if 'elements' not in color and len(color['elements']) < 3:
                return False
    
        return True

def parse_payload(data: Dict) -> (List[str], ColorData):

    missions = data.get('missions', [])
    colors = data.get('colors', [])

    color_data = ColorData(
        labels=[color.get('label', 'black') for color in colors],
        rgbs=[color.get('elements', [0, 0, 0]) for color in colors]
    )

    return missions, color_data

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

    payload_str = request.form.get('payload')
    if not payload_str:
        return "No payload found"

    try:
        payload = json.loads(payload_str)
    except:
        return "Invalid payload. Payload must be json format"

    if not validation(payload):
        return "Invalid payload. Payload must be following format. { missions: [], colors: [{ label: 'red', elements: [255, 0, 0] }] }}"

    missions, color_data = parse_payload(payload)

    print(missions)
    print(color_data)

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

    print(response)

    return json.dumps(response)

    # io_buf = io.BytesIO(buffer)
    # return io_buf.getvalue()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
