import torch
import clip
from typing import List
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageChecker:

    def __init__(self):
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.model = model
        self.preprocess = preprocess

    def check(self, file_path: str, caption_list: List[str]) -> (str, float):
        image = self.preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        text = clip.tokenize(caption_list).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            max_score = probs[0][0]
            selected_caption = caption_list[0]
            for caption, score in zip(caption_list, probs[0]):
                if max_score < score:
                    max_score = score
                    selected_caption = caption

            return selected_caption, max_score