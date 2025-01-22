from clip import CLIP
from custom_dataset import CustomDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

dataset = CustomDataset()
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=20, prefetch_factor=10, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=20, prefetch_factor=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
cnt = 0
best_loss = float('inf')
progress_bar = tqdm(range(100))
patience = 10
for epoch in progress_bar:
    model.train()
    for img, token in tqdm(train_loader):
        img = img.to(device)
        seq = token['input_ids'].to(device)
        mask = token['attention_mask'].to(device)
        optimizer.zero_grad()
        loss = model((img, seq, mask))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, token in val_loader:
            img = img.to(device)
            seq = token['input_ids'].to(device)
            mask = token['attention_mask'].to(device)
            loss = model((img, seq, mask))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './best_model.pth')
    else:
        cnt += 1
        print('patience : ', patience)
    if cnt == patience:
        progress_bar.set_postfix(loss=val_loss)
        break
    progress_bar.update()
