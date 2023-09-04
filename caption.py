import open_clip
import torch
from PIL import Image
import time

start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load an AI model.
model, _, transform = open_clip.create_model_and_transforms(
    'coca_ViT-L-14',
    pretrained='mscoco_finetuned_laion2B-s13B-b90k',
    device=device,
)

# file_path = 'images/sample.jpg'
file_path = 'outputs/result.jpg'
img = Image.open(file_path).convert('RGB')

im = transform(img).unsqueeze(0).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    generated = model.generate(im, seq_len=20, num_beam_groups=1)

caption = (
    open_clip.decode(generated[0].detach())
    .split('<end_of_text>')[0]
    .replace('<start_of_text>', '')
)

end = time.time()

print(end - start)

print(caption)