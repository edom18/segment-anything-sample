import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from typing import List
from segment_anything import SamPredictor, sam_model_registry

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Supported device is {device}')

class SAM:
    def __init__(self):
        self.model = sam_model_registry['default'](checkpoint='models/sam_vit_h_4b8939.pth')
        self.model.to(device=device)

    def crop(self, file_path: str) -> np.array:

        if not file_path:
            return None

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ih, iw = image.shape[-2:]

        input_point = np.array([[ih / 2, iw / 2]])
        input_label = np.array([1])
        predictor = SamPredictor(self.model)
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        print(masks.shape)

        most_idx = -1
        max_score = 0
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if max_score < score:
                max_score = score
                most_idx = i

        print(f'Found index is {most_idx}, score is {max_score}')

        mask = masks[most_idx]
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * image

        gs_mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(
            gs_mask_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 輪郭（Bounding box）でクロップ
        x, y, w, h = cv2.boundingRect(contours[0])

        result = image[y:y+h, x:x+w]

        return result