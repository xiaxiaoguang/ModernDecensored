import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import re
from tqdm import tqdm
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file 

# --- CONFIGURATION ---
TRAIN_MODE = "bar" 
MODEL_PATH = "./waifu_inpaint_xl_local"
BASE_DATASET_ROOT = "./dataset_refined"
RESOLUTION = 1024
BATCH_SIZE = 1

# [CRITICAL] Tuned for Small Dataset (300 images)
LEARNING_RATE = 5e-6
EPOCHS = 5  # STOP EARLY. Do not go to 15.
RANK = 16    # Low rank prevents memorization
ALPHA = 8    # Alpha < Rank = More stable, less aggressive
DROPOUT = 0.1 # Randomly disable neurons to force robust learning

TRIGGER_WORD = "reconstruct, genital detail, high quality, uncensored"
# DATASET_DIR = os.path.join(BASE_DATASET_ROOT, f"inpainter_{TRAIN_MODE}")
OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"

class RobustInpaintDataset(Dataset):
    def __init__(self, root_dir, resolution=1024, mode="hybrid", augment_mask=True):
        """
        Args:
            mode (str): Options are 'bar', 'mosaic', or 'hybrid'
        """
        self.root_dir = root_dir
        self.augment_mask = augment_mask
        self.res = resolution

        if mode == "bar":
            self.sub_tasks = ['inpainter_bar']
        elif mode == "mosaic":
            self.sub_tasks = ['inpainter_mosaic']
        else:
            self.sub_tasks = ['inpainter_bar', 'inpainter_mosaic']
            
        print(f"[Dataset] Initialized with Mode: {mode.upper()}")
        
        self.image_entries = [] # Stores tuple: (subfolder_name, filename)

        # Iterate only through selected subfolders
        for sub in self.sub_tasks:
            sub_gt_path = os.path.join(root_dir, sub, "ground_truth")
            
            if not os.path.exists(sub_gt_path):
                print(f"[Dataset] Warning: {sub_gt_path} not found. Skipping.")
                continue
                
            files = [f for f in os.listdir(sub_gt_path) if f.endswith('.png')]
            
            # Simple deduplication: Keep all files
            for f in files:
                self.image_entries.append((sub, f))
        
        print(f"[Dataset] Total Images: {len(self.image_entries)}")

        # [CRITICAL] Do NOT remove Normalize. It is required for Stable Diffusion.
        self.img_tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Transforms [0,1] -> [-1,1]
        ])
        
        # Mask must be resized with NEAREST to keep edges sharp
        self.resize = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST)
        self.crop = transforms.CenterCrop(resolution)

    def __len__(self): 
        return len(self.image_entries)

    def __getitem__(self, idx):
        sub_folder, fn = self.image_entries[idx]
        
        gt_path = os.path.join(self.root_dir, sub_folder, "ground_truth", fn)
        censored_path = os.path.join(self.root_dir, sub_folder, "censored", fn)
        mask_path = os.path.join(self.root_dir, sub_folder, "mask", fn)

        gt = Image.open(gt_path).convert("RGB")
        censored = Image.open(censored_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment_mask and random.random() > 0.3:
            kernel_size = random.randrange(1, 6, 2)
            if random.random() > 0.5:
                mask = mask.filter(ImageFilter.MaxFilter(kernel_size)) 
            else:
                mask = mask.filter(ImageFilter.MinFilter(kernel_size)) 

        return {
            "pixel_values": self.img_tf(gt),
            "mask_values": transforms.ToTensor()(self.crop(self.resize(mask))),
            "masked_image_values": self.img_tf(censored),
            "filename": fn
        }
    
def save_compatible_lora(unet, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state_dict = unet.state_dict()
    peft_dict = {k: v for k, v in state_dict.items() if "lora" in k}
    diffusers_dict = {}
    for k, v in peft_dict.items():
        new_k = k.replace("base_model.model.", "")
        if "lora_A" in new_k: new_k = new_k.replace("lora_A", "lora.down")
        elif "lora_B" in new_k: new_k = new_k.replace("lora_B", "lora.up")
        new_k = re.sub(r"\.default(_\d+)?", "", new_k)
        if not new_k.startswith("unet."): new_k = f"unet.{new_k}"
        diffusers_dict[new_k] = v
    save_file(diffusers_dict, os.path.join(output_dir, "pytorch_lora_weights.safetensors"))
    
def visualize_results(model_path, lora_folder, dataset, output_path, num_samples=3):
    print(f"\n[Visual Test] Generating Validation Samples...")
    torch.cuda.empty_cache()
    try:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        pipe.load_lora_weights(lora_folder, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora()
    except Exception as e:
        print(f"[Visual Test] Error loading pipeline: {e}")
        return

    os.makedirs(output_path, exist_ok=True)
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        # 1. Get the filename and subfolder info from the dataset
        # Note: We need to access the internal list to know the subfolder
        sub_folder, filename = dataset.image_entries[idx]
        
        # 2. Determine Strength based on censorship type
        # Bar censorship needs high strength (0.99) to completely overwrite the black pixels
        # Mosaic censorship needs lower strength (0.66) to guide the generation using the underlying colors
        if "mosaic" in sub_folder:
            denoising_strength = 0.66
            print(f"  > Processing {filename} (Mosaic) -> Strength: 0.66")
        else:
            denoising_strength = 0.99 
            print(f"  > Processing {filename} (Bar) -> Strength: 0.99")

        # 3. Load Images
        full_censored_path = os.path.join(dataset.root_dir, sub_folder, "censored", filename)
        full_mask_path = os.path.join(dataset.root_dir, sub_folder, "mask", filename)
        full_gt_path = os.path.join(dataset.root_dir, sub_folder, "ground_truth", filename)

        censored = Image.open(full_censored_path).convert("RGB").resize((1024, 1024))
        mask = Image.open(full_mask_path).convert("L").resize((1024, 1024))
        gt = Image.open(full_gt_path).convert("RGB").resize((1024, 1024))
        
        # 4. Generate
        result = pipe(
            prompt="reconstruct, genital detail, high quality, uncensored, lineart, manga style",
            negative_prompt="mosaic, black bars, censor, error, blurry, low quality",
            image=censored,
            mask_image=mask,
            num_inference_steps=35, 
            guidance_scale=7.5,
            strength=denoising_strength 
        ).images[0]
        
        w, h = censored.size
        grid = Image.new("RGB", (w * 4, h))
        grid.paste(censored, (0, 0))
        grid.paste(mask.convert("RGB"), (w, 0))
        grid.paste(gt, (w * 2, 0))
        grid.paste(result, (w * 3, 0))
        grid.save(os.path.join(output_path, f"val_{filename}_s{int(denoising_strength*100)}.png"))
        
    del pipe
    torch.cuda.empty_cache()
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
    # train()
    # TRAIN_MODE = "bar" 
    # OUTPUT_DIR = f"./output_lora_{TRAIN_MODE}"
    # train()
    chk_path=f"./output_lora_{TRAIN_MODE}/checkpoint-2"
    dataset = RobustInpaintDataset(BASE_DATASET_ROOT, RESOLUTION, TRAIN_MODE, augment_mask=False)
    visualize_results(MODEL_PATH, chk_path, dataset, os.path.join(chk_path, "visuals"),num_samples=20)
