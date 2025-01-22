import torch
from tqdm import tqdm
import os
from PIL import Image
import albumentations as A
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='/media/sien/media/data/Imagenet10/imagenet-10/'):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.path_list = []
        self.prompt = []
        for root, dirs, files in tqdm(os.walk(data_path)):
            dir = os.path.basename(root)
            for file in files:
                self.path_list.append(os.path.join(root, file))
                self.prompt.append(f'There is a {dir}.')
        model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                #A.Normalize(max_pixel_value=255.0),
            ]
        )

    def __getitem__(self, idx):
        prompt = self.prompt[idx]
        img = Image.open(self.path_list[idx]).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)['image']

        token = self.tokenizer(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30
        )

        return torch.from_numpy(img).permute(2,0,1)/224.0, token

    def __len__(self):
        return len(self.prompt)


class TestCustomDataset(TestCase):
    def test_dataset(self):
        dataset = CustomDataset(data_path='/media/sien/media/data/Imagenet10/imagenet-10/')
        img, prompt = next(iter(dataset))
        plt.imshow(img.to('cpu').permute(1,2,0).numpy())
        plt.show()
        print(prompt)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=20)
        loader = iter(loader)
        while True:
            try:
                next(loader)
            except StopIteration:
                break
