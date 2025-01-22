import torch
from torch import nn
import torch.nn.functional as F
from md import IMGEncoder, TextEncoder, ProjectionHead

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.img_encoder = IMGEncoder()
        self.img_projection = ProjectionHead(2048)
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(768)
        self.temperature = 0.8

    def forward(self, batch):
        img, prompt, attn_mask = batch
        prompt = prompt.squeeze(1)

        img_emb = self.img_encoder(img)
        img_emb = self.img_projection(img_emb)
        txt_emb = self.text_encoder(prompt, attn_mask)
        txt_emb = self.text_projection(txt_emb)

        img_emb = F.normalize(img_emb, p=2, dim=-1)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        logits_per_image = img_emb @ txt_emb.T
        logits_per_text = txt_emb @ img_emb.T

        batch_size = img_emb.size(0)
        labels = torch.arange(batch_size, device=img_emb.device)

        loss_img = F.cross_entropy(logits_per_image / self.temperature, labels)
        loss_txt = F.cross_entropy(logits_per_text / self.temperature, labels)
        loss = (loss_img + loss_txt) / 2
        return loss

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    model = CLIP()
    loss = model((images, input_ids, attention_mask))
    print(loss)
