import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset, DataLoader
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model
# Get the parent directory of the current file and add it to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.datasets import RobustInpaintDataset,save_compatible_lora,visualize_results

# --- CONFIGURATION ---
TRAIN_MODE = "bar" 
MODEL_PATH = "./waifu_inpaint_xl_local"
BASE_DATASET_ROOT = "./dataset_refined"
RESOLUTION = 1024
BATCH_SIZE = 1

# [CRITICAL] Tuned for Small Dataset (450 images)
LEARNING_RATE = 1e-6
EPOCHS = 10  # STOP EARLY. Do not go to 15.
RANK = 32    # Low rank prevents memorization
ALPHA = 32    # Alpha < Rank = More stable, less aggressive
DROPOUT = 0.1 # Randomly disable neurons to force robust learning

TRIGGER_WORD = "reconstruct, genital detail, high quality, uncensored"
# DATASET_DIR = os.path.join(BASE_DATASET_ROOT, f"inpainter_{TRAIN_MODE}")
OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"

def train():
    device = "cuda"
    weight_dtype = torch.float16

    # 1. Load Models
    # VAE MUST BE FLOAT32 for accurate encoding
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet").to(device, dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    tokenizer_two = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(device, weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2").to(device, weight_dtype)
    
    # 2. Setup LoRA (Regularized)
    lora_config = LoraConfig(
        r=RANK, 
        lora_alpha=ALPHA, 
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "ff.net.0.proj", "ff.net.2"],
        lora_dropout=DROPOUT,
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    # 3. Pre-compute Text Embeddings
    def get_embeds(prompt):
        with torch.no_grad():
            inputs = [tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device) 
                      for tokenizer in [tokenizer_one, tokenizer_two]]
            prompt_embeds_1 = text_encoder_one(inputs[0], output_hidden_states=True).hidden_states[-2]
            enc_out_2 = text_encoder_two(inputs[1], output_hidden_states=True)
            prompt_embeds_2 = enc_out_2.hidden_states[-2]
            pooled_embeds = enc_out_2[0]
            full_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        return full_prompt_embeds, pooled_embeds

    prompt_embeds, pooled_embeds = get_embeds(TRIGGER_WORD)
    add_time_ids = torch.tensor([RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]).to(device, weight_dtype).unsqueeze(0)

    # 4. Dataset with Augmentation
    dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    snr_gamma = 5.0
    print(f"[*] Training Start | Rank: {RANK} | Images: {len(dataset)} | Augmentation: ON")

    for epoch in range(EPOCHS):
        unet.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            with torch.no_grad():
                # STRICT VAE SCALING (Do not remove!)
                latents = vae.encode(batch["pixel_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(batch["masked_image_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)
                masked_latents = masked_latents.to(dtype=weight_dtype)
                # Resize mask to latent size (1/8th resolution)
                mask = F.interpolate(batch["mask_values"].to(device, weight_dtype), size=(RESOLUTION // 8, RESOLUTION // 8))

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(weight_dtype)
            # 9-CHANNEL INPUT
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            model_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids}
            ).sample

            # Min-SNR Loss
            snr = (noise_scheduler.alphas_cumprod[timesteps] / (1 - noise_scheduler.alphas_cumprod[timesteps]))
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / (snr + 1e-5)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validate every epoch
        chk_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
        save_compatible_lora(unet, chk_path)
        visualize_results(MODEL_PATH, chk_path, dataset, os.path.join(chk_path, "visuals"))

if __name__ == "__main__":
    train()
    # TRAIN_MODE = "bar" 
    # OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"
    # train()
    # chk_path=f"./output_lora_{TRAIN_MODE}/checkpoint-3"
    # dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=False)
    # visualize_results(MODEL_PATH, chk_path, dataset, os.path.join(chk_path, "visuals"),num_samples=20)
