"""
Generative model architectures for cross-family identity collapse benchmark:
1. Optimal Transport Conditional Flow Matching (OT-CFM, Ours, 6.97M)
2. Denoising Diffusion Probabilistic Model (DDPM, 0.97M)
3. Conditional Variational Autoencoder (CVAE, 16.18M)
4. Auxiliary Classifier GAN (ACGAN, 8.12M)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES, NUM_GENERA, NULL_SPECIES_ID, NULL_GENUS_ID


# ==============================================================================
# 1. OT-CFM Generator (Ours, Flow Matching, 6.97M)
# ==============================================================================
class BlurPool2D(nn.Module):
    """Anti-aliased strided downsampling (Zhang ICML 2019)."""
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.channels = channels
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
        self.register_buffer('kernel', k[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.channels)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        style = self.proj(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 2*C, 1, 1)
        gamma, beta = style.chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


class ResBlock2D(nn.Module):
    """Residual 2D Convolution block with FiLM class conditioning."""
    def __init__(self, in_chans: int, out_chans: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_chans)
        self.film = FiLMBlock(out_chans, cond_dim)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_chans)
        self.shortcut = nn.Conv2d(in_chans, out_chans, kernel_size=1) if in_chans != out_chans else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.film(h, cond)
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class SelfAttention2D(nn.Module):
    """Global self-attention for 2D spatial feature maps with learned 2D positional embedding."""
    def __init__(self, channels: int, num_heads: int = 4, h_grid: int = 32, w_grid: int = 64):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.pos_emb = nn.Parameter(torch.randn(1, channels, h_grid, w_grid) * 0.02)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x) + self.pos_emb[:, :, :H, :W]
        h_flat = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        qkv = self.qkv(h_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.out_proj(out).permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


class InsectSpectrogramGenerator(nn.Module):
    """
    Class-Conditional Spectrogram Flow-Matching U-Net (6.97M params).
    Hierarchical species (256-d) + genus (128-d) + time (256-d) embeddings = 640-d cond.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 class_emb_dim: int = 256,
                 genus_emb_dim: int = 128,
                 time_emb_dim: int = 256,
                 base_chans: int = 64):
        super().__init__()
        self.num_classes = num_classes

        # +1 null tokens for real CFG
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)
        self.genus_emb = nn.Embedding(NUM_GENERA + 1, genus_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim

        cond_dim = class_emb_dim + genus_emb_dim + time_emb_dim

        # Encoder
        self.init_conv = nn.Conv2d(1, base_chans, kernel_size=3, padding=1)
        self.down1 = ResBlock2D(base_chans, base_chans * 2, cond_dim)
        self.blur_pool1 = BlurPool2D(base_chans * 2)
        self.down2 = ResBlock2D(base_chans * 2, base_chans * 4, cond_dim)
        self.blur_pool2 = BlurPool2D(base_chans * 4)

        # Bottleneck with global self-attention
        self.mid1 = ResBlock2D(base_chans * 4, base_chans * 4, cond_dim)
        self.bottleneck_attn = SelfAttention2D(base_chans * 4, num_heads=8, h_grid=32, w_grid=64)
        self.mid2 = ResBlock2D(base_chans * 4, base_chans * 4, cond_dim)

        # Decoder
        self.up2 = ResBlock2D(base_chans * 4 + base_chans * 4, base_chans * 2, cond_dim)
        self.up1 = ResBlock2D(base_chans * 2 + base_chans * 2, base_chans, cond_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_chans),
            nn.SiLU(),
            nn.Conv2d(base_chans, 1, kernel_size=3, padding=1),
        )

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x: torch.Tensor, class_ids: torch.Tensor,
                genus_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self._time_embedding(timesteps))
        c_emb = self.class_emb(class_ids)
        g_emb = self.genus_emb(genus_ids)
        cond  = torch.cat([t_emb, c_emb, g_emb], dim=1)

        orig_w = x.shape[-1]
        pad_w  = 256 - orig_w
        if pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, 0), mode="replicate")

        h1      = self.init_conv(x)
        h2      = self.down1(h1, cond)
        h2_pool = self.blur_pool1(h2)
        h3      = self.down2(h2_pool, cond)
        h3_pool = self.blur_pool2(h3)

        m = self.mid1(h3_pool, cond)
        m = self.bottleneck_attn(m)
        m = self.mid2(m, cond)

        m_up  = F.interpolate(m,  size=h3.shape[2:], mode="bilinear", align_corners=False)
        u2    = self.up2(torch.cat([m_up, h3], dim=1), cond)
        u2_up = F.interpolate(u2, size=h2.shape[2:], mode="bilinear", align_corners=False)
        u1    = self.up1(torch.cat([u2_up, h2], dim=1), cond)
        out   = self.out_conv(u1)

        return out[..., :orig_w]


# ==============================================================================
# 2. DDPM UNet Generator (0.97M)
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DDPMGenerator(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, time_dim: int = 256, class_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.class_emb = nn.Embedding(num_classes + 1, class_dim)
        cond_dim = time_dim + class_dim

        self.init_conv = nn.Conv2d(1, 32, 3, padding=1)
        self.cond_proj1 = nn.Linear(cond_dim, 32)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.cond_proj2 = nn.Linear(cond_dim, 64)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.cond_proj3 = nn.Linear(cond_dim, 128)

        self.mid_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t, c):
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(c)
        cond = torch.cat([t_emb, c_emb], dim=-1)

        h = F.silu(self.init_conv(x) + self.cond_proj1(cond).unsqueeze(-1).unsqueeze(-1))
        h1 = F.silu(self.down1(h) + self.cond_proj2(cond).unsqueeze(-1).unsqueeze(-1))
        h2 = F.silu(self.down2(h1) + self.cond_proj3(cond).unsqueeze(-1).unsqueeze(-1))

        h_mid = F.silu(self.mid_conv(h2))

        u2 = F.silu(self.up2(h_mid))
        u2 = F.pad(u2, (0, 1, 0, 1))

        u1 = F.silu(self.up1(u2))
        u1 = F.pad(u1, (0, 3, 0, 0))

        out = self.out_conv(u1)
        return out[:, :, :128, :235]


# ==============================================================================
# 3. CVAE Generator (16.18M)
# ==============================================================================
class CVAEGenerator(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, latent_dim: int = 128, class_emb_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(256 * 8 * 14 + class_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 14 + class_emb_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim + class_emb_dim, 256 * 8 * 14)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def encode(self, x, c):
        h = self.enc_conv(x).view(x.size(0), -1)
        c_emb = self.class_emb(c)
        hc = torch.cat([h, c_emb], dim=-1)
        return self.fc_mu(hc), self.fc_logvar(hc)

    def decode(self, z, c):
        c_emb = self.class_emb(c)
        zc = torch.cat([z, c_emb], dim=-1)
        h = self.dec_fc(zc).view(z.size(0), 256, 8, 14)
        rec = self.dec_conv(h)
        return F.pad(rec, (0, 11, 0, 0))[:, :, :128, :235]

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z, c), mu, logvar


# ==============================================================================
# 4. ACGAN Generator (8.12M)
# ==============================================================================
class ACGANGenerator(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, latent_dim: int = 128, class_emb_dim: int = 128):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)
        self.fc = nn.Linear(latent_dim + class_emb_dim, 256 * 8 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z, c):
        c_emb = self.class_emb(c)
        zc = torch.cat([z, c_emb], dim=-1)
        h = self.fc(zc).view(z.size(0), 256, 8, 14)
        rec = self.deconv(h)
        return F.pad(rec, (0, 11, 0, 0))[:, :, :128, :235]


# ==============================================================================
# Sampling Helper Functions
# ==============================================================================
@torch.inference_mode()
def sample_ot_cfm(model, class_ids, genus_ids, device, steps=30, guidance_scale=2.0):
    B = class_ids.shape[0]
    x_t = torch.randn(B, 1, 128, 235, device=device)
    dt = 1.0 / steps
    null_cls = torch.full_like(class_ids, NULL_SPECIES_ID)
    null_gen = torch.full_like(genus_ids, NULL_GENUS_ID)

    for step in range(steps):
        t_val = torch.full((B,), 1.0 - step * dt, device=device)
        v_u = model(x_t, null_cls, null_gen, t_val)
        v_c = model(x_t, class_ids, genus_ids, t_val)
        v = v_u + guidance_scale * (v_c - v_u)
        x_t = x_t - v * dt
    return x_t


@torch.inference_mode()
def sample_ddpm(model, class_ids, device, steps=30):
    B = class_ids.shape[0]
    x = torch.randn(B, 1, 128, 235, device=device)
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for t_val in reversed(range(steps)):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise_pred = model(x, t, class_ids)
        alpha = alphas[t_val]
        alpha_bar = alphas_cumprod[t_val]
        x = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
        if t_val > 0:
            x = x + betas[t_val].sqrt() * torch.randn_like(x)
    return x.clamp(-5, 5)


@torch.inference_mode()
def sample_cvae(model, class_ids, device):
    B = class_ids.shape[0]
    z = torch.randn(B, model.latent_dim, device=device)
    return model.decode(z, class_ids)


@torch.inference_mode()
def sample_acgan(model, class_ids, device):
    B = class_ids.shape[0]
    z = torch.randn(B, 128, device=device)
    return model(z, class_ids)
