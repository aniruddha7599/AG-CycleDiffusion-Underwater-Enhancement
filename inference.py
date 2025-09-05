import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import os
import math
import argparse

# =================================================================================
# 1. Configuration & Hyperparameters
# =================================================================================
class Config:
    """Configuration class for model hyperparameters and paths."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Checkpoint Path ---
    CHECKPOINT_PATH = "D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\checkpoints\checkpoint_step_60000.pth"

    # --- Image Parameters ---
    IMG_SIZE = 256

    # --- Diffusion Hyperparameters ---
    TIMESTEPS = 200

    # --- EMA Decay (for loading) ---
    EMA_DECAY = 0.999

    # --- Sampling ---
    SAMPLE_DIR = "D:\\DAIICT\\Sem 3\\Major Project 1\\AG-CycleDiffusion\\inferred_samples"
    N_STEPS = 25  # Number of sampling steps (fewer for faster inference)

cfg = Config()
os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)

# =================================================================================
# 2. Diffusion Logic & Helpers
# =================================================================================
def cosine_beta_schedule(timesteps, s=0.008, device=cfg.DEVICE):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)

class Diffusion:
    def __init__(self, timesteps=cfg.TIMESTEPS, device=cfg.DEVICE):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, n, condition_tensor, n_steps=cfg.N_STEPS):
        model.eval()
        x_t = torch.randn((n, 3, cfg.IMG_SIZE, cfg.IMG_SIZE), device=self.device)
        ts_vec = torch.linspace(self.timesteps - 1, 0, n_steps + 1).long().to(self.device)
        for i in range(n_steps):
            t = ts_vec[i].expand(n)
            pred_x0 = model(x_t, t, condition_tensor)

            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_cumprod_prev = self.alphas_cumprod[ts_vec[i + 1]].view(-1, 1, 1, 1) if ts_vec[i + 1] >= 0 else torch.ones_like(alpha_cumprod)

            pred_noise = (x_t - torch.sqrt(alpha_cumprod) * pred_x0) / torch.sqrt(1. - alpha_cumprod)
            dir_xt = torch.sqrt(1. - alpha_cumprod_prev) * pred_noise
            x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
        return (x_t.clamp(-1, 1) + 1) / 2

class EMA:
    def __init__(self, model, decay):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

# =================================================================================
# 3. Model Architectures
# =================================================================================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout_prob=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, t):
        h = self.norm(self.relu(self.conv1(x)))
        h += self.relu(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)
        return self.dropout(self.norm(self.relu(self.conv2(h))))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.down3 = Block(128, 256, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot1 = Block(256, 512, time_emb_dim)
        self.up_trans_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_1 = Block(512, 256, time_emb_dim)
        self.up_trans_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_2 = Block(256, 128, time_emb_dim)
        self.up_trans_3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_3 = Block(128, 64, time_emb_dim)
        self.out = nn.Conv2d(64, out_channels, 1)
    def forward(self, x, t, condition):
        x_cond = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t)
        h1 = self.down1(x_cond, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)
        bot = self.bot1(self.pool(h3), t_emb)
        d1 = self.up_conv_1(torch.cat([self.up_trans_1(bot), h3], dim=1), t_emb)
        d2 = self.up_conv_2(torch.cat([self.up_trans_2(d1), h2], dim=1), t_emb)
        d3 = self.up_conv_3(torch.cat([self.up_trans_3(d2), h1], dim=1), t_emb)
        return self.out(d3)

# =================================================================================
# 4. Image Loading with .NEF Support via Pillow
# =================================================================================
def load_image(image_path, transform):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load {image_path}. Ensure Pillow supports .nef (try installing 'pillow-heif'). Error: {str(e)}")
    return transform(img)

# =================================================================================
# 5. Inference
# =================================================================================
def infer(input_path, output_path, direction='water_to_land'):
    # --- Load Models ---
    U_w2l = ConditionalUNet().to(cfg.DEVICE)
    U_l2w = ConditionalUNet().to(cfg.DEVICE)

    print(f"🔄 Loading checkpoint from {cfg.CHECKPOINT_PATH}...")
    ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
    U_w2l.load_state_dict(ckpt['U_w2l'])
    U_l2w.load_state_dict(ckpt['U_l2w'])
    ema_w2l = EMA(U_w2l, cfg.EMA_DECAY)
    ema_l2w = EMA(U_l2w, cfg.EMA_DECAY)
    ema_w2l.shadow = ckpt['ema_w2l']
    ema_l2w.shadow = ckpt['ema_l2w']

    # Use EMA for sampling
    ema_U = U_w2l if direction == 'water_to_land' else U_l2w
    ema = ema_w2l if direction == 'water_to_land' else ema_l2w
    ema.copy_to(ema_U)

    diffusion = Diffusion()

    # --- Prepare Input ---
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    condition = load_image(input_path, transform).unsqueeze(0).to(cfg.DEVICE)

    # --- Sample ---
    print("📸 Generating translated image...")
    translated = diffusion.sample(ema_U, n=1, condition_tensor=condition)

    # --- Save ---
    from torchvision.utils import save_image
    save_image(translated, output_path)
    print(f"✅ Saved output to {output_path}")

if __name__ == '__main__':
    # Hardcoded input path for the specified .nef file
    input_path = "D:\\DAIICT\\Sem 3\\Major Project 1\\AG-CycleDiffusion\\raw data\\IIT Jammu Dataset\\Fish4Knowlege\\imgs\\Crowded\\400000117.jpg"
    output_path = os.path.join(cfg.SAMPLE_DIR, 'LFT_5424_translated.png')
    direction = 'water_to_land'  # Default direction; change to 'land_to_water' if needed

    infer(input_path, output_path, direction)

"""## F4K Data Generation

"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import math
import argparse
from tqdm import tqdm

# =================================================================================
# 1. Model & Diffusion Architecture (Copied from your script)
# =================================================================================

# --- Model Architectures ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout_prob=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, t):
        h = self.norm(self.relu(self.conv1(x)))
        h += self.relu(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)
        return self.dropout(self.norm(self.relu(self.conv2(h))))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.down3 = Block(128, 256, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot1 = Block(256, 512, time_emb_dim)
        self.up_trans_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_1 = Block(512, 256, time_emb_dim)
        self.up_trans_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_2 = Block(256, 128, time_emb_dim)
        self.up_trans_3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_3 = Block(128, 64, time_emb_dim)
        self.out = nn.Conv2d(64, out_channels, 1)
    def forward(self, x, t, condition):
        x_cond = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t)
        h1 = self.down1(x_cond, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)
        bot = self.bot1(self.pool(h3), t_emb)
        d1 = self.up_conv_1(torch.cat([self.up_trans_1(bot), h3], dim=1), t_emb)
        d2 = self.up_conv_2(torch.cat([self.up_trans_2(d1), h2], dim=1), t_emb)
        d3 = self.up_conv_3(torch.cat([self.up_trans_3(d2), h1], dim=1), t_emb)
        return self.out(d3)

# --- Diffusion Logic & Helpers ---
class EMA:
    def __init__(self, model, decay):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

def cosine_beta_schedule(timesteps, s=0.008, device="cuda"):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)

class Diffusion:
    def __init__(self, timesteps=200, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, n, condition_tensor, n_steps=25):
        model.eval()
        x_t = torch.randn((n, 3, 256, 256), device=self.device)
        ts_vec = torch.linspace(self.timesteps - 1, 0, n_steps + 1).long().to(self.device)
        for i in range(n_steps):
            t = ts_vec[i].expand(n)
            pred_x0 = model(x_t, t, condition_tensor)

            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_cumprod_prev = self.alphas_cumprod[ts_vec[i + 1]].view(-1, 1, 1, 1) if ts_vec[i + 1] >= 0 else torch.ones_like(alpha_cumprod)

            pred_noise = (x_t - torch.sqrt(alpha_cumprod) * pred_x0) / torch.sqrt(1. - alpha_cumprod)
            dir_xt = torch.sqrt(1. - alpha_cumprod_prev) * pred_noise
            x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
        return (x_t.clamp(-1, 1) + 1) / 2

# =================================================================================
# 2. Main Batch Inference Logic
# =================================================================================
def main(args):
    """
    Main function to run batch inference on a folder of images.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model Once ---
    print(f"🔄 Loading checkpoint from {args.checkpoint_path}...")
    U_w2l = ConditionalUNet().to(DEVICE)
    U_l2w = ConditionalUNet().to(DEVICE)

    ckpt = torch.load(args.checkpoint_path, map_location=DEVICE)

    # Select the correct model and EMA weights based on direction
    if args.direction == 'water_to_land':
        model = U_w2l
        model.load_state_dict(ckpt['U_w2l'])
        ema_weights = ckpt['ema_w2l']
    elif args.direction == 'land_to_water':
        model = U_l2w
        model.load_state_dict(ckpt['U_l2w'])
        ema_weights = ckpt['ema_l2w']
    else:
        raise ValueError("Invalid direction. Choose 'water_to_land' or 'land_to_water'.")

    # Apply EMA weights for better sample quality
    ema = EMA(model, decay=0.999) # Decay value doesn't matter here
    ema.shadow = ema_weights
    ema.copy_to(model)
    model.eval()
    print("✅ Model loaded successfully.")

    diffusion = Diffusion(device=DEVICE)

    # --- Prepare Image Transformations ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Find All Images Recursively from Specific Subfolders ---
    image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.nef')
    # Define the only folders you want to process
    target_subfolders = ['imgs', 'test imgs']

    print(f"Scanning for images in specified subfolders: {target_subfolders} within {args.input_folder}...")

    for subfolder in target_subfolders:
        # Create the full path to the subfolder to start the search
        start_path = os.path.join(args.input_folder, subfolder)

        if not os.path.isdir(start_path):
            print(f"⚠️ Warning: Subfolder '{start_path}' not found. Skipping.")
            continue

        # Walk through the target subfolder and its children
        for root, _, files in os.walk(start_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("❌ No images found. Check your input folder and specified subfolders.")
        return

    print(f" found {len(image_paths)} images to process.")

    # --- Process Each Image ---
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # --- Determine Output Path and Create Directories ---
            relative_path = os.path.relpath(os.path.dirname(img_path), args.input_folder)
            output_dir = os.path.join(args.output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            filename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{filename}.png")

            # --- Prepare Input Image ---
            condition_image = Image.open(img_path).convert("RGB")
            condition = transform(condition_image).unsqueeze(0).to(DEVICE)

            # --- Generate Translated Image ---
            translated = diffusion.sample(model, n=1, condition_tensor=condition, n_steps=args.n_steps)

            # --- Save Output ---
            from torchvision.utils import save_image
            save_image(translated, output_path)

        except Exception as e:
            print(f"⚠️ Could not process {img_path}. Error: {e}")
            continue

    print("\n🎉 Batch inference complete!")
    print(f" results saved to: {args.output_folder}")

if __name__ == '__main__':
    # --- 1. DEFINE YOUR PATHS HERE ---
    # IMPORTANT: Use an 'r' before the quotes for Windows paths
    input_folder = r"D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\raw data\IIT Jammu Dataset\Fish4Knowlege"
    output_folder = r"D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\inferred_samples"
    checkpoint_path = r"D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\checkpoints\checkpoint_step_60000.pth"

    # --- 2. DO NOT CHANGE THE CODE BELOW ---
    parser = argparse.ArgumentParser(description="Batch inference script for AG-CycleDiffusion.")

    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the root folder containing the target subfolders (e.g., "imgs", "test imgs").')

    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the folder where translated images will be saved.')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint .pth file.')

    parser.add_argument('--direction', type=str, default='water_to_land',
                        choices=['water_to_land', 'land_to_water'],
                        help='Translation direction.')

    parser.add_argument('--n_steps', type=int, default=25,
                        help='Number of diffusion sampling steps.')

    # This list simulates passing arguments from the command line
    args_list = [
        '--input_folder', input_folder,
        '--output_folder', output_folder,
        '--checkpoint_path', checkpoint_path
    ]

    args = parser.parse_args(args_list)
    main(args)