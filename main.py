import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # [..., h, w]という形で画像サイズが格納されているので、それを取り出している
    h, w = mask.shape[-2:]

    # color.reshape(1, 1, -1)は
    # [1, 2, 3, 4]という配列を(1, 1, 4)次元の配列に変換
    # 具体的には[[[1, 2, 3, 4]]]という形になる
    # 最後に、boolが格納されている（0, 1）のmaskに色をかけることで、
    # マスクが存在しているところだけに色を塗っている
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Supported device is {device}')

sam = sam_model_registry['default'](checkpoint='models/sam_vit_h_4b8939.pth')
sam.to(device=device)

file_path = 'images/sample.jpg'

image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_point = np.array([[500, 875]])
input_label = np.array([1])
predictor = SamPredictor(sam)
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

cv2.imwrite('outputs/result.jpg', result)

plt.imshow(result)
plt.show()