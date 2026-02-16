import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import re
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
from accelerate import Accelerator # <--- Added for Multi-GPU
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
BATCH_SIZE = 1 # Effective batch size will be BATCH_SIZE * Num GPUs

# [CRITICAL] Tuned for Small Dataset (500 images)
LEARNING_RATE = 2e-6
EPOCHS = 10
RANK = 16   
ALPHA = 8
DROPOUT = 0.1 

TRIGGER_WORD = "reconstruct, genital detail, high quality, uncensored"
OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"
    
def train():
    # 1. Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    weight_dtype = torch.float16

    # 2. Load Models
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet").to(device, dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    tokenizer_two = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(device, weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2").to(device, weight_dtype)
    
    # 3. Setup DoRA 
    lora_config = LoraConfig(
        r=RANK, 
        lora_alpha=ALPHA, 
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "ff.net.0.proj", "ff.net.2"],
        lora_dropout=DROPOUT,
        use_dora=True, 
    )
    unet = get_peft_model(unet, lora_config)
    
    # [CRITICAL] Enable Gradient Checkpointing to stop OOM errors
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

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

    dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Prepare everything through Accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    snr_gamma = 5.0
    
    if accelerator.is_main_process:
        print(f"[*] Training Start | Rank: {RANK} | DoRA: ON | Checkpointing: ON | GPUs: {accelerator.num_processes}")

    for epoch in range(EPOCHS):
        unet.train()
        
        # Only show progress bar on the main GPU
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=not accelerator.is_main_process)
        
        for batch in pbar:
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(batch["masked_image_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)
                masked_latents = masked_latents.to(dtype=weight_dtype)
                mask = F.interpolate(batch["mask_values"].to(device, weight_dtype), size=(RESOLUTION // 8, RESOLUTION // 8))

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(weight_dtype)
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            model_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids}
            ).sample

            snr = (noise_scheduler.alphas_cumprod[timesteps] / (1 - noise_scheduler.alphas_cumprod[timesteps]))
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / (snr + 1e-5)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            # 5. Backward pass through Accelerator
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
            optimizer.step()
            optimizer.zero_grad()
            
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 6. Synchronize and Save/Validate only on Main Process
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            chk_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
            # Safely unwrap the model before saving
            unwrapped_unet = accelerator.unwrap_model(unet)
            save_compatible_lora(unwrapped_unet, chk_path)
            visualize_results(MODEL_PATH, chk_path, dataset, os.path.join(chk_path, "visuals"))

if __name__ == "__main__":
    train()