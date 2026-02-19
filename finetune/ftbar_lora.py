import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset, DataLoader, Subset
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

from utils.datasets import RobustInpaintDataset,save_compatible_lora,generate_validation_grid

# --- CONFIGURATION ---
TRAIN_MODE = "bar" 
MODEL_PATH = "./waifu_inpaint_xl_local"
BASE_DATASET_ROOT = "./dataset_refined"
RESOLUTION = 1024
BATCH_SIZE = 1

# [CRITICAL] Tuned for Small Dataset (450 images)
LEARNING_RATE = 1e-6
EPOCHS = 10  # STOP EARLY. Do not go to 15.
RANK = 48    # Low rank prevents memorization
ALPHA = 8    # Alpha < Rank = More stable, less aggressive
DROPOUT = 0.1 # Randomly disable neurons to force robust learning

TRIGGER_WORD = "reconstruct, genital detail, high quality, uncensored"
# DATASET_DIR = os.path.join(BASE_DATASET_ROOT, f"inpainter_{TRAIN_MODE}")
OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"

def train():
    device = "cuda"
    weight_dtype = torch.float16
    
    # --- GRADIENT ACCUMULATION SETUP ---
    GRADIENT_ACCUMULATION_STEPS = 4 # Adjust this to simulate your desired batch size (e.g., batch_size 1 * 4 steps = effective batch size 4)

    # 1. Load Models (Unchanged)
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet").to(device, dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    tokenizer_two = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(device, weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2").to(device, weight_dtype)
    
    # 2. Setup LoRA (Unchanged)
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

    # 3. Pre-compute Text Embeddings (Unchanged)
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

    # 4. Dataset Setup (Unchanged)
    base_train_dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=True)
    base_val_dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=False)
    
    total_size = len(base_train_dataset)
    train_size = int(0.95 * total_size)
    
    indices = torch.randperm(total_size).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    train_dataset = Subset(base_train_dataset, train_idx)
    val_dataset = Subset(base_val_dataset, val_idx)
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    snr_gamma = 5.0
    
    print(f"[*] Training Start | Rank: {RANK} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Grad Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(EPOCHS):
        unet.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Ensure gradients are zeroed out before starting the epoch
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            with torch.no_grad():
                # STRICT VAE SCALING
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

            # 1. Base MSE Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) 
            
            # 2. Min-SNR Loss Weights
            snr = (noise_scheduler.alphas_cumprod[timesteps] / (1 - noise_scheduler.alphas_cumprod[timesteps]))
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / (snr + 1e-5)
            
            # Apply Min-SNR to loss
            loss = (loss * mse_loss_weights).mean()

            # 3. Scale Loss for Gradient Accumulation
            # We divide the loss so the accumulated gradients average out correctly over the steps
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            # 4. Step Optimizer ONLY when accumulation steps are reached or at the end of the epoch
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Display unscaled loss for accurate monitoring in the progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- VALIDATION EVALUATION LOOP ---
        unet.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                b_size = val_batch["pixel_values"].shape[0]
                
                latents = vae.encode(val_batch["pixel_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(val_batch["masked_image_values"].to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                latents, masked_latents = latents.to(dtype=weight_dtype), masked_latents.to(dtype=weight_dtype)
                
                mask = F.interpolate(val_batch["mask_values"].to(device, weight_dtype), size=(RESOLUTION // 8, RESOLUTION // 8))
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(weight_dtype)
                
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                
                b_prompt = prompt_embeds.expand(b_size, -1, -1)
                b_pooled = pooled_embeds.expand(b_size, -1)
                b_time = add_time_ids.expand(b_size, -1)

                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=b_prompt,
                    added_cond_kwargs={"text_embeds": b_pooled, "time_ids": b_time}
                ).sample

                val_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\n[*] Epoch {epoch+1} Metrics | Validation Loss: {avg_val_loss:.4f}")

        # --- SAVING & VISUALIZATION ---
        chk_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
        save_compatible_lora(unet, chk_path)
        
        generate_validation_grid(
            val_dataset=val_dataset,
            unet=unet,
            vae=vae,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            noise_scheduler=noise_scheduler,
            epoch=epoch,
            output_dir=OUTPUT_DIR,
            device=device,
            weight_dtype=weight_dtype
        )
        
if __name__ == "__main__":
    train()
    # TRAIN_MODE = "bar" 
    # OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"
    # train()
    # chk_path=f"./output_lora_{TRAIN_MODE}/checkpoint-6"
    # dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=False)
    # visualize_results(MODEL_PATH, chk_path, dataset, os.path.join(chk_path, "visuals"), num_samples=20)