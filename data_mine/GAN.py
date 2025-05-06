import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import clip
from PIL import Image
import os
import numpy as np

# === Hyperparameters ===
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
NOISE_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128

# 保存路径
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load CLIP model ===
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# === Generator ===
class Generator(nn.Module):
    def __init__(self, noise_dim, text_dim, img_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + text_dim, 256 * 8 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def forward(self, z, text_embed):
        x = torch.cat([z, text_embed], dim=1)
        x = self.fc(x).view(-1, 256, 8, 8)
        return self.deconv(x)

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, text_dim, img_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*8*8 + text_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embed):
        x = self.conv(img).view(img.size(0), -1)
        x = torch.cat([x, text_embed], dim=1)
        return self.fc(x)

# === Helper: Encode text with CLIP ===
def encode_texts(texts):
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# === Placeholder Dataset (to be replaced with real loader) ===
class DummyTextImageDataset(torch.utils.data.Dataset):
    def __init__(self, n=500):
        self.texts = ["a modern office", "a sunny garden", "a wooden cabin", "a concrete hallway"] * (n // 4)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        z = torch.randn(NOISE_DIM)
        return text, z

# === Main Training ===
G = Generator(NOISE_DIM, 512).to(DEVICE)
D = Discriminator(512).to(DEVICE)
g_opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_opt = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCELoss()

dataset = DummyTextImageDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for texts, z in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        z = z.to(DEVICE)
        text_embed = encode_texts(texts).to(DEVICE)

        # === Train Discriminator ===
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        # Fake images
        fake_imgs = G(z, text_embed)

        # D loss
        d_real = D(fake_imgs.detach(), text_embed)
        d_fake = D(torch.randn_like(fake_imgs), text_embed)
        d_loss = criterion(d_real, fake_labels) + criterion(d_fake, real_labels)

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # === Train Generator ===
        d_pred = D(fake_imgs, text_embed)
        g_loss = criterion(d_pred, real_labels)

        # + CLIP similarity loss (encourage generated image to match text)
        clip_imgs = nn.functional.interpolate(fake_imgs, size=(224, 224))
        image_features = clip_model.encode_image(clip_imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        clip_loss = 1 - torch.cosine_similarity(image_features, text_embed, dim=-1).mean()

        total_g_loss = g_loss + clip_loss

        g_opt.zero_grad()
        total_g_loss.backward()
        g_opt.step()

    # 保存 Generator 和 Discriminator 的参数
    torch.save(G.state_dict(), os.path.join(SAVE_DIR, f"generator_epoch.pth"))
    torch.save(D.state_dict(), os.path.join(SAVE_DIR, f"discriminator_epoch.pth"))
    # 保存中间图像
    os.makedirs("./output", exist_ok=True)
    save_image(fake_imgs[::10], f"output/sample_epoch_{epoch+1}.png", normalize=True)

print("Training completed.")
