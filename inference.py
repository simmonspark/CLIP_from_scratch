import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
from clip import CLIP
from custom_dataset import CustomDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIP().to(device)
model.load_state_dict(torch.load('./best_model.pth', map_location=device))
model.eval()

dataset = CustomDataset()
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=20, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=20)

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

query_text = 'there is a penguin'
encoded_query = tokenizer(
    text=query_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=30
).to(device)

with torch.no_grad():
    text_query = model.text_encoder(
        encoded_query['input_ids'],
        encoded_query['attention_mask']
    )
    text_query_proj = model.text_projection(text_query)
    text_distribution = F.softmax(text_query_proj, dim=-1)

fig, axes = plt.subplots(9, 9, figsize=(12, 12))
axes = axes.flatten()
count = 0

for img, tok in train_loader:
    store = img.clone()
    img = img.to(device)
    with torch.no_grad():
        img_key = model.img_encoder(img)
        img_key_proj = model.img_projection(img_key)
        img_distribution = F.log_softmax(img_key_proj, dim=-1)
        cross_entropy = - (text_distribution * img_distribution).sum(dim=-1)
        best_idx = torch.argmin(cross_entropy).item()

    best_img = store[best_idx].permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    best_img = (best_img * std) + mean
    best_img = np.clip(best_img, 0, 1)

    axes[count].imshow(best_img)
    axes[count].axis('off')
    count += 1

    if count == 81:
        break

plt.tight_layout()
plt.show()
