import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import os
from tqdm import tqdm
import math
import itertools
import random

# For saving generated images and AMP
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast

# =================================================================================
# 1. Configuration & Hyperparameters
# =================================================================================
class Config:
    """Configuration class for model hyperparameters and paths."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Checkpointing & Resuming ---
    RESUME_CHECKPOINT_PATH = "D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\checkpoints\checkpoint_step_10000.pth"

    # --- Paths ---
    LAND_IMG_PATH = r"D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\Land"
    WATER_IMG_PATH = r"D:\DAIICT\Sem 3\Major Project 1\AG-CycleDiffusion\Underwater"

    # --- Training Hyperparameters ---
    IMG_SIZE = 256
    BATCH_SIZE = 1
    TOTAL_ITERS = 60000
    SEED = 42

    # --- Optimizers ---
    LR_U = 2e-4
    LR_D = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999

    # --- Scheduler Hyperparameters ---
    WARMUP_ITERS = 2000

    # --- Loss Weights ---
    LAMBDA_CYCLE = 10.0
    LAMBDA_PERCEPTUAL = 1.0
    LAMBDA_GAN = 1.0

    # --- Phased Training Curriculum (in iterations) ---
    PHASE1_ITERS = 30000  # Cycle + Perceptual pretraining
    PHASE2_RAMP_ITERS = 10000 # Steps to ramp up GAN loss in Phase 2

    # --- Diffusion Hyperparameters ---
    TIMESTEPS = 200

    # --- Training Mechanics ---
    USE_AMP = True
    EMA_DECAY = 0.999

    # --- Checkpointing & Sampling ---
    CHECKPOINT_DIR = "checkpoints"
    SAMPLE_DIR = "samples"
    SAVE_CHECKPOINT_STEP = 2000
    SAVE_IMAGE_STEP = 500

cfg = Config()
os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    if cfg.DEVICE == "cuda":
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = True

set_seed(cfg.SEED)


# =================================================================================
# 2. Diffusion Logic & Helpers
# =================================================================================
def cosine_beta_schedule(timesteps, s=0.008, device=cfg.DEVICE):
    steps = timesteps + 1; x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x/timesteps) + s) / (1 + s) * torch.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)

class Diffusion:
    def __init__(self, timesteps=cfg.TIMESTEPS, device=cfg.DEVICE):
        self.timesteps, self.device = timesteps, device
        self.betas = cosine_beta_schedule(timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def noise_images(self, x0, t):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise, noise

    @torch.no_grad()
    def sample(self, model, n, condition_tensor, n_steps=25):
        model.eval()
        x_t = torch.randn((n, 3, cfg.IMG_SIZE, cfg.IMG_SIZE), device=self.device)
        ts_vec = torch.linspace(self.timesteps-1, 0, n_steps+1).long().to(self.device)
        for i in range(n_steps):
            t = ts_vec[i].expand(n)
            pred_x0 = model(x_t, t, condition_tensor)

            alpha_cumprod = self.alphas_cumprod[t].view(-1,1,1,1)
            alpha_cumprod_prev = self.alphas_cumprod[ts_vec[i+1]].view(-1,1,1,1) if ts_vec[i+1] >= 0 else torch.ones_like(alpha_cumprod)

            pred_noise = (x_t - torch.sqrt(alpha_cumprod) * pred_x0) / torch.sqrt(1. - alpha_cumprod)
            dir_xt = torch.sqrt(1. - alpha_cumprod_prev) * pred_noise
            x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
        model.train()
        return (x_t.clamp(-1,1)+1)/2

class EMA:
    def __init__(self, model, decay):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items(): self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)
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
        self.relu, self.norm, self.dropout = nn.ReLU(), nn.GroupNorm(8, out_ch), nn.Dropout(dropout_prob)
    def forward(self, x, t):
        h = self.norm(self.relu(self.conv1(x)))
        h += self.relu(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)
        return self.dropout(self.norm(self.relu(self.conv2(h))))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000)/ (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:,None] * embeddings[None,:]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp=nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim,time_emb_dim), nn.ReLU())
        self.down1, self.down2, self.down3 = Block(in_channels,64,time_emb_dim), Block(64,128,time_emb_dim), Block(128,256,time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot1 = Block(256,512,time_emb_dim)
        self.up_trans_1, self.up_conv_1 = nn.ConvTranspose2d(512,256,2,2), Block(512,256,time_emb_dim)
        self.up_trans_2, self.up_conv_2 = nn.ConvTranspose2d(256,128,2,2), Block(256,128,time_emb_dim)
        self.up_trans_3, self.up_conv_3 = nn.ConvTranspose2d(128,64,2,2), Block(128,64,time_emb_dim)
        self.out = nn.Conv2d(64,out_channels,1)
    def forward(self, x, t, condition):
        x_cond = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t)
        h1 = self.down1(x_cond, t_emb); h2 = self.down2(self.pool(h1), t_emb); h3 = self.down3(self.pool(h2), t_emb)
        bot = self.bot1(self.pool(h3), t_emb)
        d1 = self.up_conv_1(torch.cat([self.up_trans_1(bot), h3], dim=1), t_emb)
        d2 = self.up_conv_2(torch.cat([self.up_trans_2(d1), h2], dim=1), t_emb)
        d3 = self.up_conv_3(torch.cat([self.up_trans_3(d2), h1], dim=1), t_emb)
        return self.out(d3)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(i,o,n=True): return [nn.Conv2d(i,o,4,2,1), nn.InstanceNorm2d(o) if n else nn.Identity(), nn.LeakyReLU(0.2,True)]
        self.model=nn.Sequential(*block(in_channels,64,False),*block(64,128),*block(128,256),*block(256,512),nn.Conv2d(512,1,4,1,1))
    def forward(self, img): return self.model(img)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(cfg.DEVICE).eval()
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])
        for param in self.feature_extractor.parameters(): param.requires_grad=False
        self.l1, self.normalize = nn.L1Loss(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def forward(self, gen, real):
        gen_r, real_r = (gen.clamp(-1,1)+1)*0.5, (real.clamp(-1,1)+1)*0.5
        gen_res, real_res = F.interpolate(gen_r, (224,224)), F.interpolate(real_r, (224,224))
        return self.l1(self.feature_extractor(self.normalize(gen_res)), self.feature_extractor(self.normalize(real_res)))

# =================================================================================
# 4. Data Loading
# =================================================================================
class ImageDataset(Dataset):
    def __init__(self, root_water, root_land, transform=None):
        self.transform = transform
        self.files_water = sorted([f for f in os.listdir(root_water) if f.endswith(('.png','.jpg','.jpeg'))])
        self.files_land = sorted([f for f in os.listdir(root_land) if f.endswith(('.png','.jpg','.jpeg'))])
        self.root_water, self.root_land = root_water, root_land
    def __getitem__(self, index):
        img_w = Image.open(os.path.join(self.root_water, self.files_water[index % len(self.files_water)])).convert("RGB")
        img_l = Image.open(os.path.join(self.root_land, self.files_land[random.randint(0, len(self.files_land)-1)])).convert("RGB")
        return {"water": self.transform(img_w), "land": self.transform(img_l)}
    def __len__(self): return max(len(self.files_water), len(self.files_land))

# =================================================================================
# 5. Training Loop
# =================================================================================
def train():
    # --- Initialization ---
    U_w2l, U_l2w = ConditionalUNet().to(cfg.DEVICE), ConditionalUNet().to(cfg.DEVICE)
    D_land, D_water = Discriminator().to(cfg.DEVICE), Discriminator().to(cfg.DEVICE)
    ema_w2l, ema_l2w = EMA(U_w2l, cfg.EMA_DECAY), EMA(U_l2w, cfg.EMA_DECAY)
    diffusion = Diffusion()

    optimizer_U = torch.optim.AdamW(itertools.chain(U_w2l.parameters(), U_l2w.parameters()), lr=cfg.LR_U, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_D = torch.optim.AdamW(itertools.chain(D_land.parameters(), D_water.parameters()), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    crit_GAN, crit_cycle, crit_perc = nn.MSELoss(), nn.L1Loss(), PerceptualLoss()
    scaler_U, scaler_D = GradScaler(enabled=cfg.USE_AMP), GradScaler(enabled=cfg.USE_AMP)

    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataloader = DataLoader(ImageDataset(cfg.WATER_IMG_PATH, cfg.LAND_IMG_PATH, transform=transform), batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    sched_U = torch.optim.lr_scheduler.SequentialLR(optimizer_U, [torch.optim.lr_scheduler.LinearLR(optimizer_U, 1e-6, 1.0, cfg.WARMUP_ITERS), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_U, cfg.TOTAL_ITERS - cfg.WARMUP_ITERS)], [cfg.WARMUP_ITERS])
    sched_D = torch.optim.lr_scheduler.SequentialLR(optimizer_D, [torch.optim.lr_scheduler.LinearLR(optimizer_D, 1e-6, 1.0, cfg.WARMUP_ITERS), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, cfg.TOTAL_ITERS - cfg.WARMUP_ITERS)], [cfg.WARMUP_ITERS])

    start_iter = 0
    if cfg.RESUME_CHECKPOINT_PATH:
        print(f"🔄 Resuming training from {cfg.RESUME_CHECKPOINT_PATH}...")
        ckpt = torch.load(cfg.RESUME_CHECKPOINT_PATH, map_location=cfg.DEVICE)
        U_w2l.load_state_dict(ckpt['U_w2l']); U_l2w.load_state_dict(ckpt['U_l2w'])
        D_land.load_state_dict(ckpt['D_land']); D_water.load_state_dict(ckpt['D_water'])
        optimizer_U.load_state_dict(ckpt['opt_U']); optimizer_D.load_state_dict(ckpt['opt_D'])
        sched_U.load_state_dict(ckpt['sched_U']); sched_D.load_state_dict(ckpt['sched_D'])
        scaler_U.load_state_dict(ckpt['scaler_U']); scaler_D.load_state_dict(ckpt['scaler_D'])
        ema_w2l.shadow = ckpt['ema_w2l']; ema_l2w.shadow = ckpt['ema_l2w']
        start_iter = ckpt['iter'] + 1
        print(f"✅ Resumed successfully from iteration {start_iter}.")

    print("🚀 Starting Simplified Training for AG-CycleDiffusion...")
    pbar = tqdm(range(start_iter, cfg.TOTAL_ITERS), initial=start_iter, total=cfg.TOTAL_ITERS)
    data_iter = iter(dataloader)

    for step in pbar:
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(dataloader); batch = next(data_iter)
        real_water, real_land = batch["water"].to(cfg.DEVICE), batch["land"].to(cfg.DEVICE)

        # --- Determine Training Phase ---
        phase = 1; lambda_gan_current = 0.0
        if step >= cfg.PHASE1_ITERS:
            phase = 2
            lambda_gan_current = min(1.0, (step - cfg.PHASE1_ITERS) / cfg.PHASE2_RAMP_ITERS) * cfg.LAMBDA_GAN

        # --- Train U-Nets (Generators) ---
        optimizer_U.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.USE_AMP):
            t = torch.randint(0, cfg.TIMESTEPS, (real_water.size(0),), device=cfg.DEVICE).long()

            # -- Symmetrical Cycles: Predict clean images directly --
            noisy_land, _ = diffusion.noise_images(real_land, t)
            fake_land = U_w2l(noisy_land, t, real_water)

            noisy_fake_land, _ = diffusion.noise_images(fake_land, t)
            reconstructed_water = U_l2w(noisy_fake_land, t, fake_land)

            noisy_water, _ = diffusion.noise_images(real_water, t)
            fake_water = U_l2w(noisy_water, t, real_land)

            noisy_fake_water, _ = diffusion.noise_images(fake_water, t)
            reconstructed_land = U_w2l(noisy_fake_water, t, fake_water)

            # -- Cycle & Perceptual Loss --
            loss_cycle = crit_cycle(reconstructed_water, real_water) + crit_cycle(reconstructed_land, real_land)
            loss_perceptual = crit_perc(reconstructed_water, real_water) + crit_perc(reconstructed_land, real_land)

            # -- Total U-Net Loss Calculation --
            loss_U = cfg.LAMBDA_CYCLE * loss_cycle + cfg.LAMBDA_PERCEPTUAL * loss_perceptual

            if lambda_gan_current > 0:
                pred_fake_land = D_land(fake_land)
                loss_GAN_w2l = crit_GAN(pred_fake_land, torch.ones_like(pred_fake_land))

                pred_fake_water = D_water(fake_water)
                loss_GAN_l2w = crit_GAN(pred_fake_water, torch.ones_like(pred_fake_water))
                loss_U += lambda_gan_current * (loss_GAN_w2l + loss_GAN_l2w)

        scaler_U.scale(loss_U).backward()
        scaler_U.step(optimizer_U); scaler_U.update()
        ema_w2l.update(U_w2l); ema_l2w.update(U_l2w)

        # --- Train Discriminators ---
        if lambda_gan_current > 0:
            optimizer_D.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.USE_AMP):
                pred_real_land = D_land(real_land)
                pred_fake_land = D_land(fake_land.detach())
                loss_D_land = (crit_GAN(pred_real_land, torch.ones_like(pred_real_land)) + crit_GAN(pred_fake_land, torch.zeros_like(pred_fake_land))) / 2

                pred_real_water = D_water(real_water)
                pred_fake_water = D_water(fake_water.detach())
                loss_D_water = (crit_GAN(pred_real_water, torch.ones_like(pred_real_water)) + crit_GAN(pred_fake_water, torch.zeros_like(pred_fake_water))) / 2
                total_loss_D = (loss_D_land + loss_D_water) / 2

            scaler_D.scale(total_loss_D).backward()
            scaler_D.step(optimizer_D); scaler_D.update()
        else:
            total_loss_D = torch.tensor(0.0, device=cfg.DEVICE)

        sched_U.step(); sched_D.step()
        pbar.set_postfix({"U Loss": loss_U.item(), "D Loss": total_loss_D.item(), "LR": optimizer_U.param_groups[0]['lr'], "Phase": phase})

        # --- Sampling and Checkpointing ---
        if (step + 1) % cfg.SAVE_IMAGE_STEP == 0:
            print("📸 Sampling images...")
            ema_U_w2l_sample = ConditionalUNet().to(cfg.DEVICE)
            ema_w2l.copy_to(ema_U_w2l_sample)
            ema_U_w2l_sample.eval()
            sampled_land = diffusion.sample(ema_U_w2l_sample, n=1, condition_tensor=real_water[:1])
            img_sample = torch.cat((real_water[:1].add(1).mul(0.5), sampled_land), 0)
            save_image(img_sample, f"{cfg.SAMPLE_DIR}/step_{step+1}.png", nrow=1)
            del ema_U_w2l_sample

        if (step + 1) % cfg.SAVE_CHECKPOINT_STEP == 0:
             print(f"💾 Saving checkpoint for step {step+1}...")
             torch.save({
                'iter': step, 'U_w2l': U_w2l.state_dict(), 'U_l2w': U_l2w.state_dict(),
                'D_land': D_land.state_dict(), 'D_water': D_water.state_dict(),
                'opt_U': optimizer_U.state_dict(), 'opt_D': optimizer_D.state_dict(),
                'sched_U': sched_U.state_dict(), 'sched_D': sched_D.state_dict(),
                'scaler_U': scaler_U.state_dict(), 'scaler_D': scaler_D.state_dict(),
                'ema_w2l': ema_w2l.shadow, 'ema_l2w': ema_l2w.shadow
            }, f"{cfg.CHECKPOINT_DIR}/checkpoint_step_{step+1}.pth")

if __name__ == '__main__':
    train()