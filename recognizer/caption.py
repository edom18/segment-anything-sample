import torch
import clip
from typing import List
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Caption:

    def __init__(self):
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.model = model
        self.preprocess = preprocess

    def generate(self, caption_list: List[str]):
        image = self.preprocess(Image.open('images/sample2.jpg')).unsqueeze(0).to(device)
        text = clip.tokenize(caption_list).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print(f'Label probs: {probs}')