import torch
import open_clip
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load an AI model.
model, _, transform = open_clip.create_model_and_transforms(
    'coca_ViT-L-14',
    pretrained='mscoco_finetuned_laion2B-s13B-b90k',
    device=device,
)

text = open_clip.tokenize(['a dog', 'a human', 'the park'])
file_path = 'images/sample.jpg'
img = Image.open(file_path).convert('RGB')

im = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    image_feature = model.encode_image(im)
    text_feature = model.encode_text(text)

    l, t = model(image, text)
    probs = l.softmax(dim=-1).cpu().numpy()