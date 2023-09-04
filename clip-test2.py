import torch
import clip
from PIL import Image
import time

start = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open('images/sample2.jpg')).unsqueeze(0).to(device)
text = clip.tokenize(['a dog', 'a human', 'under the bench', 'wood', 'on the grass']).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

end = time.time()
print(end - start)

print(f'Label probs: {probs}')