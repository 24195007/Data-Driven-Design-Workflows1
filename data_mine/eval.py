import torch
import clip
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import torch.nn as nn
# === Hyperparameters ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_DIM = 100
TEXT_DIM = 512
IMAGE_SIZE = 128
SAVE_DIR = "checkpoints"

# === Load CLIP model ===
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# === Generator Model Definition ===
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

# === Helper: Encode text with CLIP ===
def encode_texts(texts):
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# === Load the trained Generator ===
G = Generator(NOISE_DIM, TEXT_DIM).to(DEVICE)
G.load_state_dict(torch.load(os.path.join(SAVE_DIR, "generator_epoch.pth")))
G.eval()

# === Function to generate an image from text ===
def generate_image_from_text(text):
    # Encode the text to obtain the embedding
    text_embed = encode_texts([text]).to(DEVICE)

    # Generate random noise vector (latent space)
    z = torch.randn(1, NOISE_DIM).to(DEVICE)

    # Generate the image
    with torch.no_grad():
        generated_image = G(z, text_embed)

    # Convert the generated image to a format that can be saved
    generated_image = (generated_image + 1) / 2  # Normalize to [0, 1] range
    save_image(generated_image, f"generated_image.png", normalize=True)
    return generated_image

# === Example Usage ===
text = "a sunny garden"
generated_image = generate_image_from_text(text)
generated_image = generated_image.squeeze().cpu().numpy().transpose(1, 2, 0)
generated_image = (generated_image * 255).astype(np.uint8)
Image.fromarray(generated_image).show()  # Show the generated image
