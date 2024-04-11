import os                                                                     # nopep8
import sys                                                                    # nopep8
import torch                                                                  # nopep8
from PIL import Image                                                         # nopep8
model_dir = '/project/ai/zhouyi_compass/Model_Resource/github/openai/CLIP'    # nopep8
sys.path.append(model_dir)                                                    # nopep8
import clip                                                                   # nopep8

model_path = os.path.join(model_dir, 'ViT-B-32.pt')
image_path = os.path.join(model_dir, 'CLIP.png')
output_dir = os.path.join(model_dir, 'OUTPUTS/TORCH')
os.makedirs(output_dir, mode=0o775, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(model_path, device=device, jit=True)
model.eval()

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(['a diagram', 'a dog', 'a cat']).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print('Label probs of original model:', probs)  # prints: [[0.9927937  0.00421068 0.00299572]

pt_path = os.path.join(output_dir, 'torch_clip_vit_b_32.pt')
model.save(pt_path)
traced_model = torch.jit.load(pt_path)

with torch.no_grad():
    image_features2 = traced_model.encode_image(image)
    text_features2 = traced_model.encode_text(text)
    logits_per_image2, logits_per_text2 = traced_model(image, text)
    probs2 = logits_per_image2.softmax(dim=-1).cpu().numpy()

print('Label probs of traced model:', probs2)  # prints: [[0.9927937  0.00421068 0.00299572]
