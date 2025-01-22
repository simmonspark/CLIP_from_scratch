import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertTokenizer
import unittest

class IMGEncoder(nn.Module):
    def __init__(self):
        super(IMGEncoder, self).__init__()
        self.model = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=256,
            dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        model_name = "distilbert-base-uncased"
        self.model = DistilBertModel.from_pretrained(model_name)
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class test(unittest.TestCase):
    def test_text_encoder(self):
        model = TextEncoder()
        tmp = torch.randint(0, 10, (5, 10))
        attn = torch.ones_like(tmp)
        out = model(tmp, attn)
        print(out.shape)